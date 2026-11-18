// SPDX-License-Identifier: LGPL-2.1-or-later
/*
 * libonnxruntime_ane.so: an ONNX Runtime plugin execution provider for the Apple Neural Engine,
 * through libane. ONNX Runtime 1.24 or later loads it (RegisterExecutionProviderLibrary); its EP
 * is "AneExecutionProvider" on an NPU device of its own, as ONNX Runtime finds no PCI identity
 * for apple-ane's node.
 *
 * It runs EPContext nodes (domain com.microsoft) whose source is "ane": a program Apple's
 * compiler built, a .hwx, held in the node (embed_mode 1, its bytes in ep_cache_context) or
 * beside the model (embed_mode 0, ep_cache_context a path inside the model's directory, or
 * inside the session's ep.context_file_path one for a model loaded from memory; opened beneath
 * it, so neither ".." nor a symlink leads out, and a regular file only). The node's
 * inputs and outputs are the program's, in the order of their buffer index, unless the INTS
 * attributes ane_src_index and ane_dst_index give each one's index. Their declared shapes give
 * each buffer's rows (README.md, "The engine's buffers"); float32 and float16 tensors are taken
 * and given, and the engine computes in float16. ort/ane-epctx.py writes such a model from
 * anehwx's model.hwx and model.json.
 *
 * Session options: ep.AneExecutionProvider.device (the node, else libane's choice) and
 * ep.AneExecutionProvider.timeout_ms (a run's, 3000 by default).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <linux/openat2.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "onnxruntime_c_api.h"
#include "../ane.h"

/* the ONNX Runtime API it asks for, 1.24's, whatever its headers' (newer ones run it too) */
#define ANE_ORT_API 24
#if ORT_API_VERSION < ANE_ORT_API
#error "ONNX Runtime's headers are older than 1.24"
#endif

#define EP_NAME	  "AneExecutionProvider"
#define EP_VENDOR "libane"
/* Apple's PCI vendor ID: ONNX Runtime's policies pair a factory with devices of its vendor */
#define EP_VENDOR_ID 0x106B
#define EXPORT	     __attribute__((visibility("default")))

static const OrtApi *api;
static const OrtEpApi *ep_api;

struct factory {
	OrtEpFactory base;
	OrtHardwareDevice *hw; /* ours, kept until the factory goes */
};

struct kernel;

struct ep {
	OrtEp base;
	struct ane *dev;
	char *ctx_dir;	 /* ep.context_file_path's directory, or NULL */
	int timeout_ms;	 /* 0: libane's */
	bool ctx_enable; /* ep.context_enable: a context model asked for, which it cannot write */
	pthread_mutex_t lock;
	struct kernel *kernels; /* every compiled node, freed with the EP at the latest */
};

/* one of a node's inputs or outputs, in the fused node's order */
struct io {
	char *name;
	int buf;			/* libane's input or output number */
	size_t src;			/* an input's: the fused node's input it is fed from */
	ONNXTensorElementDataType type; /* float or float16 */
	int64_t dims[8];
	size_t ndims;
	size_t rows, width,
		stride; /* the buffer's layout: rows of width halves, stride bytes apart */
};

struct kernel {
	OrtNodeComputeInfo base;
	struct ep *ep;
	struct kernel *next;
	char *name;
	struct ane_nn *nn;
	size_t nin, nout;
	struct io *in, *out;
	pthread_mutex_t lock; /* a program's buffers serve one run at a time */
};

static OrtStatus *fail(OrtErrorCode code, const char *fmt, ...)
	__attribute__((format(printf, 2, 3)));

static OrtStatus *fail(OrtErrorCode code, const char *fmt, ...)
{
	char msg[512];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(msg, sizeof(msg), fmt, ap);
	va_end(ap);
	return api->CreateStatus(code, msg);
}

/* a libane error as a status */
static OrtStatus *ane_fail(int err, const char *what)
{
	return fail(err == -EINVAL ? ORT_INVALID_ARGUMENT : ORT_EP_FAIL, "%s: %s", what,
		    strerror(-err));
}

#define TRY(call)                                                                                  \
	do {                                                                                       \
		OrtStatus *st_ = (call);                                                           \
		if (st_)                                                                           \
			return st_;                                                                \
	} while (0)

/* ---- the factory ---- */

static const char *factory_name(const OrtEpFactory *f)
{
	(void)f;
	return EP_NAME;
}

static const char *factory_vendor(const OrtEpFactory *f)
{
	(void)f;
	return EP_VENDOR;
}

static uint32_t factory_vendor_id(const OrtEpFactory *f)
{
	(void)f;
	return EP_VENDOR_ID;
}

static const char *factory_version(const OrtEpFactory *f)
{
	(void)f;
	return ANE_VERSION;
}

static OrtStatus *factory_devices(OrtEpFactory *base, const OrtHardwareDevice *const *devices,
				  size_t num_devices, OrtEpDevice **ep_devices, size_t max,
				  size_t *num)
{
	struct factory *f = (struct factory *)base;
	struct ane *dev;
	(void)devices;
	(void)num_devices;

	*num = 0;
	if (max < 1 || ane_open(&dev, NULL) < 0)
		return NULL; /* no engine here: no device, and no error */
	ane_close(dev);
	if (!f->hw)
		TRY(ep_api->CreateHardwareDevice(OrtHardwareDeviceType_NPU, EP_VENDOR_ID, 0,
						 "Apple", NULL, &f->hw));
	TRY(ep_api->CreateEpDevice(base, f->hw, NULL, NULL, &ep_devices[0]));
	*num = 1;
	return NULL;
}

/* CPU memory only: no allocator, no data transfer, no streams */
static OrtStatus *factory_allocator(OrtEpFactory *f, const OrtMemoryInfo *info,
				    const OrtKeyValuePairs *opts, OrtAllocator **out)
{
	(void)f;
	(void)info;
	(void)opts;
	*out = NULL;
	return NULL;
}

static void factory_release_allocator(OrtEpFactory *f, OrtAllocator *a)
{
	(void)f;
	(void)a;
}

static OrtStatus *factory_transfer(OrtEpFactory *f, OrtDataTransferImpl **out)
{
	(void)f;
	*out = NULL;
	return NULL;
}

static bool factory_stream_aware(const OrtEpFactory *f)
{
	(void)f;
	return false;
}

static OrtStatus *factory_stream(OrtEpFactory *f, const OrtMemoryDevice *md,
				 const OrtKeyValuePairs *opts, OrtSyncStreamImpl **out)
{
	(void)f;
	(void)md;
	(void)opts;
	*out = NULL;
	return NULL;
}

/* ---- the EP ---- */

static const char *ep_name(const OrtEp *e)
{
	(void)e;
	return EP_NAME;
}

/* a STRING attribute's bytes, NUL-terminated (it may hold binary data: *len is its size) */
static OrtStatus *attr_string(const OrtNode *node, const char *name, char **out, size_t *len)
{
	const OrtOpAttr *a = NULL;
	OrtStatus *st;
	size_t n = 0;

	*out = NULL;
	if (len)
		*len = 0;
	st = api->Node_GetAttributeByName(node, name, &a);
	if (st || !a) { /* an unknown or unset attribute: none */
		if (st)
			api->ReleaseStatus(st);
		return NULL;
	}
	st = api->ReadOpAttr(a, ORT_OP_ATTR_STRING, NULL, 0, &n);
	if (st)
		api->ReleaseStatus(st); /* a size probe fails and gives the size */
	*out = malloc(n + 1);
	if (!*out)
		return fail(ORT_FAIL, "out of memory");
	if (n) {
		st = api->ReadOpAttr(a, ORT_OP_ATTR_STRING, *out, n, &n);
		if (st) {
			free(*out);
			*out = NULL;
			return st;
		}
	}
	(*out)[n] = 0;
	if (len)
		*len = n;
	return NULL;
}

static OrtStatus *attr_int(const OrtNode *node, const char *name, int64_t def, int64_t *out)
{
	const OrtOpAttr *a = NULL;
	OrtStatus *st;
	size_t n;

	*out = def;
	st = api->Node_GetAttributeByName(node, name, &a);
	if (st || !a) {
		if (st)
			api->ReleaseStatus(st);
		return NULL;
	}
	return api->ReadOpAttr(a, ORT_OP_ATTR_INT, out, sizeof(*out), &n);
}

/* an INTS attribute: *count -1 when it is absent */
static OrtStatus *attr_ints(const OrtNode *node, const char *name, int64_t **out, ssize_t *count)
{
	const OrtOpAttr *a = NULL;
	OrtStatus *st;
	size_t n = 0;

	*out = NULL;
	*count = -1;
	st = api->Node_GetAttributeByName(node, name, &a);
	if (st || !a) {
		if (st)
			api->ReleaseStatus(st);
		return NULL;
	}
	st = api->ReadOpAttr(a, ORT_OP_ATTR_INTS, NULL, 0, &n);
	if (st)
		api->ReleaseStatus(st);
	*out = malloc(n ? n : 1);
	if (!*out)
		return fail(ORT_FAIL, "out of memory");
	if (n) {
		st = api->ReadOpAttr(a, ORT_OP_ATTR_INTS, *out, n, &n);
		if (st) {
			free(*out);
			*out = NULL;
			return st;
		}
	}
	*count = (ssize_t)(n / sizeof(int64_t));
	return NULL;
}

/* whether NODE is an EPContext node of ours */
static OrtStatus *is_ours(const OrtNode *node, bool *ours)
{
	const char *op, *domain;
	char *source = NULL;
	int64_t main_context;
	OrtStatus *st;

	*ours = false;
	TRY(api->Node_GetOperatorType(node, &op));
	TRY(api->Node_GetDomain(node, &domain));
	if (strcmp(op, "EPContext") != 0 || strcmp(domain, "com.microsoft") != 0)
		return NULL;
	TRY(attr_int(node, "main_context", 1, &main_context));
	st = attr_string(node, "source", &source, NULL);
	if (st)
		return st;
	if (main_context == 1 && source != NULL && strcmp(source, "ane") == 0)
		*ours = true;
	free(source);
	return NULL;
}

static OrtStatus *ep_capability(OrtEp *e, const OrtGraph *graph, OrtEpGraphSupportInfo *info)
{
	OrtNodeFusionOptions opts = { .ort_version_supported = ANE_ORT_API };
	const OrtNode **nodes;
	OrtStatus *st = NULL;
	size_t n, i;
	bool ours;
	(void)e;

	TRY(api->Graph_GetNumNodes(graph, &n));
	if (!n)
		return NULL;
	nodes = (const OrtNode **)calloc(n, sizeof(*nodes));
	if (!nodes)
		return fail(ORT_FAIL, "out of memory");
	st = api->Graph_GetNodes(graph, nodes, n);
	for (i = 0; !st && i < n; i++) {
		st = is_ours(nodes[i], &ours);
		/* each node a program: one fused node each */
		if (!st && ours)
			st = ep_api->EpGraphSupportInfo_AddNodesToFuse(info, &nodes[i], 1, &opts);
	}
	free((void *)nodes);
	return st;
}

/* REL, ep_cache_context, as a path inside the model's directory: relative, with no ".." */
static OrtStatus *check_relative(const char *rel)
{
	const char *p;

	if (!*rel || rel[0] == '/')
		return fail(ORT_INVALID_GRAPH, "ane: ep_cache_context '%s' is not a relative path",
			    rel);
	for (p = rel; p; p = strchr(p, '/'), p = p ? p + 1 : NULL)
		if (!strncmp(p, "..", 2) && (p[2] == '/' || !p[2]))
			return fail(ORT_INVALID_GRAPH,
				    "ane: ep_cache_context '%s' leaves the model's directory", rel);
	return NULL;
}

/*
 * Read the program REL inside DIR into *BUF: opened beneath DIR (no symlink out of it), a
 * regular file (a FIFO does not block the open)
 */
static OrtStatus *read_program(const char *dir, const char *rel, void **buf, size_t *len)
{
	struct open_how how = {
		.flags = O_RDONLY | O_NONBLOCK | O_CLOEXEC,
		.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS,
	};
	OrtStatus *st = NULL;
	char *data = NULL;
	struct stat sb;
	size_t got = 0;
	int dfd, fd;

	*buf = NULL;
	TRY(check_relative(rel));
	dfd = open(dir, O_PATH | O_DIRECTORY | O_CLOEXEC);
	if (dfd < 0)
		return ane_fail(-errno, dir);
	fd = (int)syscall(SYS_openat2, dfd, rel, &how, sizeof(how));
	if (fd < 0) {
		int e = errno;

		close(dfd);
		if (e == EXDEV)
			return fail(ORT_INVALID_GRAPH,
				    "ane: ep_cache_context '%s' leaves the model's directory", rel);
		return ane_fail(-e, rel);
	}
	close(dfd);
	if (fstat(fd, &sb))
		st = ane_fail(-errno, rel);
	else if (!S_ISREG(sb.st_mode) || sb.st_size <= 0)
		st = fail(ORT_INVALID_GRAPH, "ane: %s is not a program file", rel);
	if (!st) {
		data = malloc((size_t)sb.st_size);
		if (!data)
			st = fail(ORT_FAIL, "out of memory");
	}
	while (!st && data && got < (size_t)sb.st_size) {
		ssize_t r = read(fd, data + got, (size_t)sb.st_size - got);

		if (r < 0 && errno == EINTR)
			continue;
		if (r <= 0)
			st = r < 0 ? ane_fail(-errno, rel)
				   : fail(ORT_EP_FAIL, "ane: %s: shorter than its size", rel);
		else
			got += (size_t)r;
	}
	close(fd);
	if (st) {
		free(data);
		return st;
	}
	*buf = data;
	*len = got;
	return NULL;
}

/* the directory of PATH, a model's or a context file's */
static char *dir_of(const char *path)
{
	const char *slash = strrchr(path, '/');

	if (!slash)
		return strdup(".");
	return slash == path ? strdup("/") : strndup(path, (size_t)(slash - path));
}

/* load the program of the EPContext node NODE in GRAPH */
static OrtStatus *load_program(struct ep *ep, const OrtGraph *graph, const OrtNode *node,
			       struct ane_nn **nn)
{
	const ORTCHAR_T *model = NULL;
	char *ctx = NULL, *dir = NULL;
	void *prog = NULL;
	OrtStatus *st;
	int64_t embed;
	size_t len;
	int r;

	TRY(attr_int(node, "embed_mode", 1, &embed));
	TRY(attr_string(node, "ep_cache_context", &ctx, &len));
	if (!ctx || !len) {
		free(ctx);
		return fail(ORT_INVALID_GRAPH, "ane: an EPContext node without ep_cache_context");
	}
	if (embed) {
		r = ane_load_mem(ep->dev, ctx, len, nn);
		free(ctx);
		return r < 0 ? ane_fail(r, "ane: the node's program") : NULL;
	}
	st = api->Graph_GetModelPath(graph, &model);
	if (st) {
		api->ReleaseStatus(st);
		model = NULL;
	}
	if (model && *model) {
		dir = dir_of(model);
	} else if (ep->ctx_dir) {
		dir = strdup(ep->ctx_dir);
	} else {
		free(ctx);
		return fail(ORT_INVALID_ARGUMENT, "ane: a model from memory names its program by "
						  "path: set ep.context_file_path");
	}
	if (!dir) {
		free(ctx);
		return fail(ORT_FAIL, "out of memory");
	}
	st = read_program(dir, ctx, &prog, &len);
	free(dir);
	if (!st) {
		r = ane_load_mem(ep->dev, prog, len, nn);
		st = r < 0 ? ane_fail(r, ctx) : NULL;
	}
	free(prog);
	free(ctx);
	return st;
}

static size_t count_of(const struct io *io)
{
	return io->rows * io->width;
}

/*
 * Bind VI, the EPContext node's J-th input or output, to a program buffer: the one at program
 * index *INDEX, or the J-th with INDEX NULL. BOUND marks the buffers taken: none twice.
 */
static OrtStatus *bind_one(struct kernel *k, const OrtValueInfo *vi, bool out, const int64_t *index,
			   size_t j, bool *bound, struct io *io)
{
	int count = out ? ane_dst_count(k->nn, 0) : ane_src_count(k->nn, 0);
	const OrtTensorTypeAndShapeInfo *tsi;
	const OrtTypeInfo *ti;
	const char *name;
	ssize_t size;
	size_t d;
	int i;

	/* an omitted optional input or output (an empty name) has no value info */
	if (!vi)
		return fail(ORT_INVALID_GRAPH, "ane: the node omits its %s %zu",
			    out ? "output" : "input", j);
	TRY(api->GetValueInfoName(vi, &name));
	io->name = strdup(name);
	if (!io->name)
		return fail(ORT_FAIL, "out of memory");
	io->buf = -1;
	for (i = 0; i < count; i++) {
		int x = out ? ane_dst_index(k->nn, 0, i) : ane_src_index(k->nn, 0, i);

		if (index ? x == *index : (size_t)i == j) {
			io->buf = i;
			break;
		}
	}
	if (io->buf < 0)
		return fail(ORT_INVALID_GRAPH, "ane: %s: the program has no buffer at index %lld",
			    name, index ? (long long)*index : (long long)j);
	if (bound[io->buf])
		return fail(ORT_INVALID_GRAPH, "ane: %s: its buffer is another %s's too", name,
			    out ? "output" : "input");
	bound[io->buf] = true;
	TRY(api->GetValueInfoTypeInfo(vi, &ti));
	TRY(api->CastTypeInfoToTensorInfo(ti, &tsi));
	if (!tsi)
		return fail(ORT_INVALID_GRAPH, "ane: %s is not a tensor", name);
	TRY(api->GetTensorElementType(tsi, &io->type));
	if (io->type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
	    io->type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
		return fail(ORT_INVALID_GRAPH, "ane: %s: float or float16 only", name);
	TRY(api->GetDimensionsCount(tsi, &io->ndims));
	if (io->ndims < 1 || io->ndims > 8)
		return fail(ORT_INVALID_GRAPH, "ane: %s: %zu axes", name, io->ndims);
	TRY(api->GetDimensions(tsi, io->dims, io->ndims));
	io->rows = 1;
	for (d = 0; d < io->ndims; d++) {
		if (io->dims[d] < 1)
			return fail(ORT_INVALID_GRAPH, "ane: %s: axis %zu is not fixed", name, d);
		if (d + 1 < io->ndims &&
		    __builtin_mul_overflow(io->rows, (size_t)io->dims[d], &io->rows))
			return fail(ORT_INVALID_GRAPH, "ane: %s: too many rows", name);
	}
	io->width = (size_t)io->dims[io->ndims - 1];
	size = out ? ane_dst_size(k->nn, 0, io->buf) : ane_src_size(k->nn, 0, io->buf);
	if (size < 0)
		return ane_fail((int)size, name);
	/* rows of width halves, each row stride bytes: rows x stride is the buffer's size */
	if ((size_t)size % io->rows || (size_t)size / io->rows / 2 < io->width)
		return fail(ORT_INVALID_GRAPH,
			    "ane: %s: a %zd-byte buffer is not %zu rows of %zu halves", name, size,
			    io->rows, io->width);
	io->stride = (size_t)size / io->rows;
	return NULL;
}

/*
 * Bind the node's value infos VI (the EPContext node's inputs or outputs, N of them) to the
 * program's buffers: io[j] is VI[j]'s, through INDEX (the program's buffer indices; NINDEX -1:
 * in the order of the index)
 */
static OrtStatus *bind(struct kernel *k, const OrtValueInfo *const *vi, size_t n, bool out,
		       const int64_t *index, ssize_t nindex, struct io *io)
{
	const char *kind = out ? "outputs" : "inputs";
	int count = out ? ane_dst_count(k->nn, 0) : ane_src_count(k->nn, 0);
	OrtStatus *st = NULL;
	bool *bound;
	size_t j;

	if (count < 0)
		return ane_fail(count, "ane: the program's buffers");
	if ((size_t)count != n)
		return fail(ORT_INVALID_GRAPH, "ane: the program has %d %s, the node %zu", count,
			    kind, n);
	if (nindex >= 0 && (size_t)nindex != n)
		return fail(ORT_INVALID_GRAPH, "ane: %zd indices for %zu %s", nindex, n, kind);
	bound = calloc(n ? n : 1, sizeof(*bound));
	if (!bound)
		return fail(ORT_FAIL, "out of memory");
	for (j = 0; !st && j < n; j++)
		st = bind_one(k, vi[j], out, nindex >= 0 ? &index[j] : NULL, j, bound, &io[j]);
	free(bound);
	return st;
}

/*
 * The fused node's input each of the node's inputs (IO, N of them) is fed from: ONNX Runtime
 * gives the fused node each tensor once, so two inputs may share one
 */
static OrtStatus *feed_from_fused(struct io *io, size_t n, const OrtValueInfo *const *vi,
				  size_t nvi)
{
	bool *fed = calloc(nvi ? nvi : 1, sizeof(*fed));
	OrtStatus *st = NULL;
	size_t j, m;

	if (!fed)
		return fail(ORT_FAIL, "out of memory");
	for (m = 0; !st && m < n; m++) {
		for (j = 0; !st && j < nvi; j++) {
			const char *name;

			st = vi[j] ? api->GetValueInfoName(vi[j], &name)
				   : fail(ORT_EP_FAIL, "ane: the fused node omits an input");
			if (!st && !strcmp(io[m].name, name))
				break;
		}
		if (!st && j == nvi)
			st = fail(ORT_EP_FAIL, "ane: the fused node has no input %s", io[m].name);
		if (!st) {
			io[m].src = j;
			fed[j] = true;
		}
	}
	for (j = 0; !st && j < nvi; j++)
		if (!fed[j])
			st = fail(ORT_EP_FAIL, "ane: the fused node's input %zu is not the node's",
				  j);
	free(fed);
	return st;
}

/*
 * The node's outputs (IO, NIO of them) in the fused node's order (VI, N of them), by name, those
 * the fused node drops (unused) left out
 */
static OrtStatus *by_fused_order(struct io *io, size_t nio, const OrtValueInfo *const *vi, size_t n,
				 struct io **out)
{
	size_t j, m;

	*out = calloc(n ? n : 1, sizeof(**out));
	if (!*out)
		return fail(ORT_FAIL, "out of memory");
	for (j = 0; j < n; j++) {
		const char *name;

		if (!vi[j])
			return fail(ORT_EP_FAIL, "ane: the fused node omits an output");
		TRY(api->GetValueInfoName(vi[j], &name));
		for (m = 0; m < nio && (!io[m].name || strcmp(io[m].name, name) != 0); m++)
			;
		if (m == nio)
			return fail(ORT_EP_FAIL, "ane: the fused node's %s is not the node's",
				    name);
		(*out)[j] = io[m];
		io[m].name = NULL; /* moved */
	}
	return NULL;
}

static void free_io(struct io *io, size_t n)
{
	size_t j;

	for (j = 0; io && j < n; j++)
		free(io[j].name);
	free(io);
}

static OrtStatus *node_values(const OrtNode *node, bool out, const OrtValueInfo ***vi, size_t *n)
{
	*vi = NULL;
	TRY(out ? api->Node_GetNumOutputs(node, n) : api->Node_GetNumInputs(node, n));
	*vi = (const OrtValueInfo **)calloc(*n ? *n : 1, sizeof(**vi));
	if (!*vi)
		return fail(ORT_FAIL, "out of memory");
	return out ? api->Node_GetOutputs(node, *vi, *n) : api->Node_GetInputs(node, *vi, *n);
}

static inline _Float16 to_half(const void *src, ONNXTensorElementDataType t, size_t i)
{
	return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? (_Float16)((const float *)src)[i]
							: ((const _Float16 *)src)[i];
}

/* input J of the fused node into its buffer, as float16 rows */
static OrtStatus *send_input(struct kernel *k, OrtKernelContext *ctx, size_t j)
{
	const struct io *io = &k->in[j];
	uint8_t *dst = ane_src_ptr(k->nn, 0, io->buf);
	OrtTensorTypeAndShapeInfo *tsi;
	ONNXTensorElementDataType t;
	const OrtValue *v;
	const void *src;
	size_t elems, r, c;
	OrtStatus *st;

	if (!dst)
		return fail(ORT_EP_FAIL, "ane: %s: no mapping", io->name);
	TRY(api->KernelContext_GetInput(ctx, io->src, &v));
	TRY(api->GetTensorTypeAndShape(v, &tsi));
	st = api->GetTensorElementType(tsi, &t);
	if (!st)
		st = api->GetTensorShapeElementCount(tsi, &elems);
	api->ReleaseTensorTypeAndShapeInfo(tsi);
	if (st)
		return st;
	if (t != io->type || elems != count_of(io))
		return fail(ORT_INVALID_ARGUMENT, "ane: %s: not the node's type or size", io->name);
	TRY(api->GetTensorData(v, &src));
	for (r = 0; r < io->rows; r++) {
		_Float16 *row = (_Float16 *)(dst + (r * io->stride));

		for (c = 0; c < io->width; c++)
			row[c] = to_half(src, t, (r * io->width) + c);
	}
	return NULL;
}

/* output J of the fused node from its buffer's float16 rows */
static OrtStatus *take_output(struct kernel *k, OrtKernelContext *ctx, size_t j)
{
	const struct io *io = &k->out[j];
	const uint8_t *src = ane_dst_ptr(k->nn, 0, io->buf);
	OrtValue *v;
	void *dst;
	size_t r, c;

	if (!src)
		return fail(ORT_EP_FAIL, "ane: %s: no mapping", io->name);
	TRY(api->KernelContext_GetOutput(ctx, j, io->dims, io->ndims, &v));
	TRY(api->GetTensorMutableData(v, &dst));
	for (r = 0; r < io->rows; r++) {
		const _Float16 *row = (const _Float16 *)(src + (r * io->stride));

		if (io->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
			memcpy((_Float16 *)dst + (r * io->width), row, io->width * sizeof(*row));
			continue;
		}
		for (c = 0; c < io->width; c++)
			((float *)dst)[(r * io->width) + c] = (float)row[c];
	}
	return NULL;
}

static OrtStatus *run_locked(struct kernel *k, OrtKernelContext *ctx)
{
	size_t j;
	int r;

	for (j = 0; j < k->nin; j++)
		TRY(send_input(k, ctx, j));
	r = ane_run(k->nn, 0, k->ep->timeout_ms);
	if (r < 0)
		return ane_fail(r, k->name);
	for (j = 0; j < k->nout; j++)
		TRY(take_output(k, ctx, j));
	return NULL;
}

static OrtStatus *kernel_compute(OrtNodeComputeInfo *base, void *state, OrtKernelContext *ctx)
{
	struct kernel *k = (struct kernel *)base;
	OrtStatus *st;
	(void)state;

	pthread_mutex_lock(&k->lock);
	st = run_locked(k, ctx);
	pthread_mutex_unlock(&k->lock);
	return st;
}

static OrtStatus *kernel_state(OrtNodeComputeInfo *base, OrtNodeComputeContext *ctx, void **state)
{
	(void)ctx;
	*state = base; /* a kernel is its node's state */
	return NULL;
}

static void kernel_release_state(OrtNodeComputeInfo *base, void *state)
{
	(void)base;
	(void)state;
}

static void kernel_free(struct kernel *k)
{
	if (!k)
		return;
	ane_free(k->nn);
	free_io(k->in, k->nin);
	free_io(k->out, k->nout);
	free(k->name);
	pthread_mutex_destroy(&k->lock);
	free(k);
}

/* the kernel of the fused node FUSED, whose graph GRAPH holds one EPContext node of ours */
static OrtStatus *compile_one(struct ep *ep, const OrtGraph *graph, const OrtNode *fused,
			      struct kernel **out)
{
	const OrtValueInfo **vin = NULL, **vout = NULL, **fin = NULL, **fout = NULL;
	size_t nvin = 0, nvout = 0, nfin = 0, nfout = 0, n;
	struct io *in = NULL, *outs = NULL;
	int64_t *sidx = NULL, *didx = NULL;
	ssize_t nsidx, ndidx;
	const OrtNode *node;
	struct kernel *k;
	const char *name;
	OrtStatus *st;
	bool ours;

	*out = NULL;
	TRY(api->Graph_GetNumNodes(graph, &n));
	if (n != 1)
		return fail(ORT_EP_FAIL, "ane: a fused graph of %zu nodes", n);
	TRY(api->Graph_GetNodes(graph, &node, 1));
	TRY(is_ours(node, &ours));
	if (!ours)
		return fail(ORT_EP_FAIL, "ane: asked to compile a node that is not ours");
	k = calloc(1, sizeof(*k));
	if (!k)
		return fail(ORT_FAIL, "out of memory");
	pthread_mutex_init(&k->lock, NULL);
	k->ep = ep;
	k->base.ort_version_supported = ANE_ORT_API;
	k->base.CreateState = kernel_state;
	k->base.Compute = kernel_compute;
	k->base.ReleaseState = kernel_release_state;
	st = api->Node_GetName(fused, &name);
	if (!st) {
		k->name = strdup(name);
		if (!k->name)
			st = fail(ORT_FAIL, "out of memory");
	}
	if (!st)
		st = load_program(ep, graph, node, &k->nn);
	if (!st && ane_procs(k->nn) != 1)
		st = fail(ORT_INVALID_GRAPH, "ane: a program of %d procedures", ane_procs(k->nn));
	if (!st)
		st = attr_ints(node, "ane_src_index", &sidx, &nsidx);
	if (!st)
		st = attr_ints(node, "ane_dst_index", &didx, &ndidx);
	if (!st)
		st = node_values(node, false, &vin, &nvin);
	if (!st)
		st = node_values(node, true, &vout, &nvout);
	if (!st) {
		in = calloc(nvin ? nvin : 1, sizeof(*in));
		outs = calloc(nvout ? nvout : 1, sizeof(*outs));
		if (!in || !outs)
			st = fail(ORT_FAIL, "out of memory");
	}
	if (!st)
		st = bind(k, vin, nvin, false, sidx, nsidx, in);
	if (!st)
		st = bind(k, vout, nvout, true, didx, ndidx, outs);
	/*
	 * ONNX Runtime's fused node may order them otherwise, gives a tensor fed to two inputs
	 * once and drops unused outputs
	 */
	if (!st)
		st = node_values(fused, false, &fin, &nfin);
	if (!st)
		st = node_values(fused, true, &fout, &nfout);
	if (!st)
		st = feed_from_fused(in, nvin, fin, nfin);
	if (!st) {
		k->in = in; /* each program input, fed from its fused input */
		k->nin = nvin;
		in = NULL;
	}
	if (!st)
		st = by_fused_order(outs, nvout, fout, nfout, &k->out);
	k->nout = k->out ? nfout : 0;
	free_io(in, nvin);
	free_io(outs, nvout);
	free((void *)vin);
	free((void *)vout);
	free((void *)fin);
	free((void *)fout);
	free(sidx);
	free(didx);
	if (st) {
		kernel_free(k);
		return st;
	}
	*out = k;
	return NULL;
}

static OrtStatus *ep_compile(OrtEp *base, const OrtGraph **graphs, const OrtNode **fused,
			     size_t count, OrtNodeComputeInfo **infos, OrtNode **ctx_nodes)
{
	struct ep *ep = (struct ep *)base;
	struct kernel *k;
	size_t i;

	for (i = 0; i < count; i++) {
		infos[i] = NULL;
		ctx_nodes[i] = NULL;
	}
	if (ep->ctx_enable && count)
		return fail(ORT_INVALID_ARGUMENT,
			    "ane: the model's ane nodes are compiled already: "
			    "ep.context_enable has no context to write");
	for (i = 0; i < count; i++) {
		OrtStatus *st = compile_one(ep, graphs[i], fused[i], &k);

		if (st || !k)
			return st;
		pthread_mutex_lock(&ep->lock);
		k->next = ep->kernels;
		ep->kernels = k;
		pthread_mutex_unlock(&ep->lock);
		infos[i] = &k->base;
	}
	return NULL;
}

static void ep_release_infos(OrtEp *base, OrtNodeComputeInfo **infos, size_t n)
{
	struct ep *ep = (struct ep *)base;
	struct kernel **p;
	size_t i;

	pthread_mutex_lock(&ep->lock);
	for (i = 0; i < n; i++)
		for (p = &ep->kernels; *p; p = &(*p)->next)
			if (&(*p)->base == infos[i]) {
				struct kernel *k = *p;

				*p = k->next;
				kernel_free(k);
				break;
			}
	pthread_mutex_unlock(&ep->lock);
}

static OrtStatus *ep_concurrent(OrtEp *e, bool *yes)
{
	(void)e;
	*yes = true; /* each program locks its own buffers */
	return NULL;
}

/* a session option, or NULL */
static char *option(const OrtSessionOptions *so, const char *key)
{
	OrtStatus *st;
	size_t n = 0;
	char *v;

	st = api->GetSessionConfigEntry(so, key, NULL, &n);
	if (st) {
		api->ReleaseStatus(st);
		if (!n)
			return NULL;
	}
	if (!n)
		return NULL;
	v = malloc(n);
	if (!v)
		return NULL;
	st = api->GetSessionConfigEntry(so, key, v, &n);
	if (st) {
		api->ReleaseStatus(st);
		free(v);
		return NULL;
	}
	return v;
}

static OrtStatus *factory_create_ep(OrtEpFactory *f, const OrtHardwareDevice *const *devices,
				    const OrtKeyValuePairs *const *meta, size_t n,
				    const OrtSessionOptions *so, const OrtLogger *logger,
				    OrtEp **out)
{
	char *node, *ctx_file, *timeout, *ctx_enable;
	OrtStatus *st = NULL;
	struct ep *ep;
	int r = 0;
	(void)f;
	(void)devices;
	(void)meta;
	(void)n;
	(void)logger;

	*out = NULL;
	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return fail(ORT_FAIL, "out of memory");
	pthread_mutex_init(&ep->lock, NULL);
	ep->base.ort_version_supported = ANE_ORT_API;
	ep->base.GetName = ep_name;
	ep->base.GetCapability = ep_capability;
	ep->base.Compile = ep_compile;
	ep->base.ReleaseNodeComputeInfos = ep_release_infos;
	ep->base.IsConcurrentRunSupported = ep_concurrent;
	node = option(so, "ep." EP_NAME ".device");
	timeout = option(so, "ep." EP_NAME ".timeout_ms");
	ctx_file = option(so, "ep.context_file_path");
	ctx_enable = option(so, "ep.context_enable");
	if (ctx_enable && strcmp(ctx_enable, "1") == 0)
		ep->ctx_enable = true;
	if (timeout) {
		char *end;
		long ms;

		errno = 0;
		ms = strtol(timeout, &end, 10);
		if (!*timeout || *end || errno || ms < 1 || ms > INT_MAX)
			st = fail(ORT_INVALID_ARGUMENT,
				  "ane: ep." EP_NAME ".timeout_ms '%s' is not a count of ms",
				  timeout);
		else
			ep->timeout_ms = (int)ms;
	}
	if (!st && ctx_file) {
		ep->ctx_dir = dir_of(ctx_file);
		if (!ep->ctx_dir)
			st = fail(ORT_FAIL, "out of memory");
	}
	if (!st) {
		r = ane_open(&ep->dev, node && *node ? node : NULL);
		if (r < 0)
			st = ane_fail(r, "ane: the device");
	}
	free(node);
	free(timeout);
	free(ctx_file);
	free(ctx_enable);
	if (st) {
		free(ep->ctx_dir);
		pthread_mutex_destroy(&ep->lock);
		free(ep);
		return st;
	}
	*out = &ep->base;
	return NULL;
}

static void factory_release_ep(OrtEpFactory *f, OrtEp *base)
{
	struct ep *ep = (struct ep *)base;
	(void)f;

	if (!ep)
		return;
	while (ep->kernels) {
		struct kernel *k = ep->kernels;

		ep->kernels = k->next;
		kernel_free(k);
	}
	ane_close(ep->dev);
	free(ep->ctx_dir);
	pthread_mutex_destroy(&ep->lock);
	free(ep);
}

/* ---- the library's entry points ---- */

// NOLINTNEXTLINE(misc-use-internal-linkage): ONNX Runtime's entry point
EXPORT OrtStatus *CreateEpFactories(const char *registered_name, const OrtApiBase *base,
				    const OrtLogger *logger, OrtEpFactory **factories, size_t max,
				    size_t *num)
{
	struct factory *f;
	(void)registered_name;
	(void)logger;

	*num = 0;
	api = base->GetApi(ANE_ORT_API);
	if (!api)
		return NULL; /* an ONNX Runtime older than 1.24: nothing to offer */
	ep_api = api->GetEpApi();
	if (max < 1)
		return fail(ORT_INVALID_ARGUMENT, "ane: no room for a factory");
	f = calloc(1, sizeof(*f));
	if (!f)
		return fail(ORT_FAIL, "out of memory");
	f->base.ort_version_supported = ANE_ORT_API;
	f->base.GetName = factory_name;
	f->base.GetVendor = factory_vendor;
	f->base.GetVendorId = factory_vendor_id;
	f->base.GetVersion = factory_version;
	f->base.GetSupportedDevices = factory_devices;
	f->base.CreateEp = factory_create_ep;
	f->base.ReleaseEp = factory_release_ep;
	f->base.CreateAllocator = factory_allocator;
	f->base.ReleaseAllocator = factory_release_allocator;
	f->base.CreateDataTransfer = factory_transfer;
	f->base.IsStreamAware = factory_stream_aware;
	f->base.CreateSyncStreamForDevice = factory_stream;
	factories[0] = &f->base;
	*num = 1;
	return NULL;
}

// NOLINTNEXTLINE(misc-use-internal-linkage): ONNX Runtime's entry point
EXPORT OrtStatus *ReleaseEpFactory(OrtEpFactory *base)
{
	struct factory *f = (struct factory *)base;

	if (f && f->hw)
		ep_api->ReleaseHardwareDevice(f->hw);
	free(f);
	return NULL;
}
