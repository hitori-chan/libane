// SPDX-License-Identifier: LGPL-2.1-or-later
/* libane (the API: ane.h), over the apple-ane uAPI (apple_ane_accel.h) and DRM's syncobjs. */
#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <drm/drm.h>
#include <linux/sync_file.h>
#include "apple_ane_accel.h"
#include "ane.h"

#define DRIVER_NAME	   "apple-ane"
#define ACCEL_DIR	   "/dev/accel"
#define DEFAULT_TIMEOUT_MS 3000

struct ane {
	int fd;
	uint64_t page;
};

struct bo {
	uint32_t handle;
	uint64_t size; /* mapped bytes, whole pages */
	void *map;
};

struct proc {
	int ready;
	int busy; /* a run that timed out may still hold the engine: the syncobj keeps its fence */
	unsigned n; /* the buffers the procedure binds */
	struct drm_apple_ane_buf_info *info;
	struct drm_apple_ane_submit_buf *sb;
	struct bo *bo;
	unsigned nsrc, ndst;
	unsigned *src, *dst; /* positions in info[] */
	uint32_t sync;
};

struct ane_nn {
	struct ane *dev;
	uint32_t program, nproc;
	struct proc *proc;
};

/* the failed call's errno, negated: never 0, so a failure never reads as success */
static int neg_errno(void)
{
	return errno > 0 ? -errno : -EIO;
}

/* an ioctl, retried as libdrm's drmIoctl() does */
static int io(int fd, unsigned long req, void *arg)
{
	int r;

	do
		r = ioctl(fd, req, arg);
	while (r && (errno == EINTR || errno == EAGAIN));
	return r ? neg_errno() : 0;
}

static int bo_create(struct ane *dev, uint64_t size, struct bo *b)
{
	struct drm_apple_ane_bo_create c = { 0 };
	int r;

	size = (size ? size : 1) + dev->page - 1;
	size &= ~(dev->page - 1);
	c.size = size;
	r = io(dev->fd, DRM_IOCTL_APPLE_ANE_BO_CREATE, &c);
	if (r)
		return r;
	b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd, (off_t)c.offset);
	if (b->map == MAP_FAILED) {
		struct drm_gem_close gc = { .handle = c.handle };

		r = neg_errno();
		b->map = NULL;
		io(dev->fd, DRM_IOCTL_GEM_CLOSE, &gc);
		return r;
	}
	b->handle = c.handle;
	b->size = size;
	return 0;
}

static void bo_destroy(struct ane *dev, struct bo *b)
{
	struct drm_gem_close gc = { .handle = b->handle };

	if (b->map)
		munmap(b->map, b->size);
	if (b->handle)
		io(dev->fd, DRM_IOCTL_GEM_CLOSE, &gc);
	memset(b, 0, sizeof(*b));
}

/* whether the DRM file FD is apple-ane's */
static int is_ane(int fd)
{
	char name[sizeof(DRIVER_NAME) + 1] = "";
	struct drm_version v = { .name = name, .name_len = sizeof(name) - 1 };

	if (io(fd, DRM_IOCTL_VERSION, &v))
		return 0;
	return v.name_len == strlen(DRIVER_NAME) && !memcmp(name, DRIVER_NAME, v.name_len);
}

/* the lowest N above PREV among the nodes /dev/accel/accelN, or -1 */
static long next_node(long prev)
{
	DIR *d = opendir(ACCEL_DIR);
	struct dirent *e;
	long best = -1;

	if (!d)
		return -1;
	while ((e = readdir(d))) {
		char *end;
		long n;

		if (strncmp(e->d_name, "accel", 5) != 0 || !isdigit((unsigned char)e->d_name[5]))
			continue;
		n = strtol(e->d_name + 5, &end, 10);
		if (!*end && n > prev && (best < 0 || n < best))
			best = n;
	}
	closedir(d);
	return best;
}

static int open_node(const char *path, int *fd)
{
	*fd = open(path, O_RDWR | O_CLOEXEC);
	if (*fd < 0)
		return neg_errno();
	if (!is_ane(*fd)) {
		close(*fd);
		*fd = -1;
		return -ENODEV;
	}
	return 0;
}

int ane_open(struct ane **out, const char *node)
{
	struct ane *dev;
	int fd = -1, r;

	*out = NULL;
	if (!node)
		node = getenv("ANE_DEVICE");
	if (node && *node) {
		r = open_node(node, &fd);
	} else {
		r = -ENODEV;
		for (long n = next_node(-1); n >= 0 && fd < 0; n = next_node(n)) {
			char path[sizeof(ACCEL_DIR "/accel") + 20];
			int e;

			snprintf(path, sizeof(path), ACCEL_DIR "/accel%ld", n);
			e = open_node(path, &fd);
			/* an apple-ane node we may not open says more than "none" */
			if (e && e != -ENODEV && r == -ENODEV)
				r = e;
		}
		if (fd >= 0)
			r = 0;
	}
	if (r)
		return r;
	dev = calloc(1, sizeof(*dev));
	if (!dev) {
		close(fd);
		return -ENOMEM;
	}
	dev->fd = fd;
	dev->page = (uint64_t)sysconf(_SC_PAGESIZE);
	*out = dev;
	return 0;
}

void ane_close(struct ane *dev)
{
	if (!dev)
		return;
	close(dev->fd);
	free(dev);
}

/* a program from SIZE bytes: copied from MEM, or read from the file FD */
static int load(struct ane *dev, const void *mem, int fd, size_t size, struct ane_nn **out)
{
	struct drm_apple_ane_program_create pc = { 0 };
	struct bo hwx = { 0 };
	struct ane_nn *nn;
	int r;

	*out = NULL;
	if (!size || size > UINT32_MAX)
		return -EINVAL;
	r = bo_create(dev, size, &hwx);
	if (r)
		return r;
	if (mem) {
		memcpy(hwx.map, mem, size);
	} else {
		for (size_t done = 0; done < size;) {
			ssize_t n = read(fd, (char *)hwx.map + done, size - done);

			if (n < 0 && errno == EINTR)
				continue;
			if (n <= 0) {
				r = n < 0 ? neg_errno() : -EIO; /* the file shrank */
				break;
			}
			done += (size_t)n;
		}
	}
	if (!r) {
		pc.handle = hwx.handle;
		pc.size = (uint32_t)size;
		r = io(dev->fd, DRM_IOCTL_APPLE_ANE_PROGRAM_CREATE, &pc);
	}
	/* the driver keeps a copy of its own */
	bo_destroy(dev, &hwx);
	if (r)
		return r;
	nn = calloc(1, sizeof(*nn));
	if (nn)
		nn->proc = calloc(pc.procedures ? pc.procedures : 1, sizeof(*nn->proc));
	if (!nn || !nn->proc) {
		struct drm_apple_ane_program_destroy pd = { .program = pc.program };

		io(dev->fd, DRM_IOCTL_APPLE_ANE_PROGRAM_DESTROY, &pd);
		free(nn);
		return -ENOMEM;
	}
	nn->dev = dev;
	nn->program = pc.program;
	nn->nproc = pc.procedures;
	*out = nn;
	return 0;
}

int ane_load_mem(struct ane *dev, const void *hwx, size_t size, struct ane_nn **out)
{
	if (!hwx) {
		*out = NULL;
		return -EINVAL;
	}
	return load(dev, hwx, -1, size, out);
}

int ane_load(struct ane *dev, const char *path, struct ane_nn **out)
{
	struct stat st;
	int fd, r;

	*out = NULL;
	fd = open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0)
		return neg_errno();
	if (fstat(fd, &st))
		r = neg_errno();
	else if (!S_ISREG(st.st_mode) || st.st_size <= 0)
		r = -EINVAL;
	else
		r = load(dev, NULL, fd, (size_t)st.st_size, out);
	close(fd);
	return r;
}

static void proc_release(struct ane *dev, struct proc *p)
{
	if (p->sync) {
		struct drm_syncobj_destroy sd = { .handle = p->sync };

		io(dev->fd, DRM_IOCTL_SYNCOBJ_DESTROY, &sd);
	}
	/* a run still going keeps its own references to the buffers and the program */
	for (unsigned i = 0; p->bo && i < p->n; i++)
		bo_destroy(dev, &p->bo[i]);
	free(p->info);
	free(p->sb);
	free(p->bo);
	free(p->src);
	free(p->dst);
	memset(p, 0, sizeof(*p));
}

void ane_free(struct ane_nn *nn)
{
	struct drm_apple_ane_program_destroy pd = { 0 };

	if (!nn)
		return;
	for (uint32_t i = 0; i < nn->nproc; i++)
		proc_release(nn->dev, &nn->proc[i]);
	pd.program = nn->program;
	io(nn->dev->fd, DRM_IOCTL_APPLE_ANE_PROGRAM_DESTROY, &pd);
	free(nn->proc);
	free(nn);
}

int ane_procs(const struct ane_nn *nn)
{
	return (int)nn->nproc;
}

/* sort the positions POS[0..N) by their buffers' index: the driver lists them in section order */
static void sort_by_index(unsigned *pos, unsigned n, const struct drm_apple_ane_buf_info *info)
{
	for (unsigned i = 1; i < n; i++) {
		unsigned k = pos[i], j = i;

		for (; j && info[pos[j - 1]].index > info[k].index; j--)
			pos[j] = pos[j - 1];
		pos[j] = k;
	}
}

/* a procedure's first use: its buffer list from the driver, a buffer object for each, a syncobj */
static int prepare(struct ane_nn *nn, int proc)
{
	struct ane *dev = nn->dev;
	struct drm_apple_ane_program_info pi = { .program = nn->program };
	struct drm_syncobj_create sc = { 0 };
	struct proc *p;
	unsigned m;
	int r;

	if (proc < 0 || (uint32_t)proc >= nn->nproc)
		return -EINVAL;
	p = &nn->proc[proc];
	if (p->ready)
		return 0;
	pi.procedure = proc;
	r = io(dev->fd, DRM_IOCTL_APPLE_ANE_PROGRAM_INFO, &pi); /* no bufs: the count only */
	if (r)
		return r;
	p->n = pi.count;
	p->nsrc = p->ndst = 0;
	m = p->n ? p->n : 1;
	p->info = calloc(m, sizeof(*p->info));
	p->sb = calloc(m, sizeof(*p->sb));
	p->bo = calloc(m, sizeof(*p->bo));
	p->src = calloc(m, sizeof(*p->src));
	p->dst = calloc(m, sizeof(*p->dst));
	if (!p->info || !p->sb || !p->bo || !p->src || !p->dst) {
		r = -ENOMEM;
		goto err;
	}
	pi.count = p->n;
	pi.bufs = (uintptr_t)p->info;
	r = io(dev->fd, DRM_IOCTL_APPLE_ANE_PROGRAM_INFO, &pi);
	if (r)
		goto err;
	if (pi.count != p->n) {
		r = -EIO;
		goto err;
	}
	for (unsigned i = 0; i < p->n; i++) {
		r = bo_create(dev, p->info[i].size, &p->bo[i]);
		if (r)
			goto err;
		p->sb[i].index = p->info[i].index;
		p->sb[i].handle = p->bo[i].handle;
		if (p->info[i].kind == DRM_APPLE_ANE_BUF_INPUT)
			p->src[p->nsrc++] = i;
		else if (p->info[i].kind == DRM_APPLE_ANE_BUF_OUTPUT)
			p->dst[p->ndst++] = i;
	}
	sort_by_index(p->src, p->nsrc, p->info);
	sort_by_index(p->dst, p->ndst, p->info);
	r = io(dev->fd, DRM_IOCTL_SYNCOBJ_CREATE, &sc);
	if (r)
		goto err;
	p->sync = sc.handle;
	p->ready = 1;
	return 0;
err:
	proc_release(dev, p);
	return r;
}

/* the position in info[] of input (DST 0) or output (DST 1) I, or a negative errno */
static int pick(struct ane_nn *nn, int proc, int dst, int i)
{
	struct proc *p;
	int r = prepare(nn, proc);

	if (r)
		return r;
	p = &nn->proc[proc];
	if (i < 0 || (unsigned)i >= (dst ? p->ndst : p->nsrc))
		return -EINVAL;
	return (int)(dst ? p->dst[i] : p->src[i]);
}

int ane_src_count(struct ane_nn *nn, int proc)
{
	int r = prepare(nn, proc);

	return r ? r : (int)nn->proc[proc].nsrc;
}

int ane_dst_count(struct ane_nn *nn, int proc)
{
	int r = prepare(nn, proc);

	return r ? r : (int)nn->proc[proc].ndst;
}

static ssize_t size_of(struct ane_nn *nn, int proc, int dst, int i)
{
	int k = pick(nn, proc, dst, i);

	return k < 0 ? k : (ssize_t)nn->proc[proc].bo[k].size;
}

ssize_t ane_src_size(struct ane_nn *nn, int proc, int i)
{
	return size_of(nn, proc, 0, i);
}

ssize_t ane_dst_size(struct ane_nn *nn, int proc, int i)
{
	return size_of(nn, proc, 1, i);
}

static int index_of(struct ane_nn *nn, int proc, int dst, int i)
{
	int k = pick(nn, proc, dst, i);

	return k < 0 ? k : (int)nn->proc[proc].info[k].index;
}

int ane_src_index(struct ane_nn *nn, int proc, int i)
{
	return index_of(nn, proc, 0, i);
}

int ane_dst_index(struct ane_nn *nn, int proc, int i)
{
	return index_of(nn, proc, 1, i);
}

void *ane_src_ptr(struct ane_nn *nn, int proc, int i)
{
	int k = pick(nn, proc, 0, i);

	return k < 0 ? NULL : nn->proc[proc].bo[k].map;
}

const void *ane_dst_ptr(struct ane_nn *nn, int proc, int i)
{
	int k = pick(nn, proc, 1, i);

	return k < 0 ? NULL : nn->proc[proc].bo[k].map;
}

int ane_send(struct ane_nn *nn, int proc, int i, const void *buf, size_t len)
{
	int k = pick(nn, proc, 0, i);

	if (k < 0)
		return k;
	if (len > nn->proc[proc].bo[k].size)
		return -EINVAL;
	memcpy(nn->proc[proc].bo[k].map, buf, len);
	return 0;
}

int ane_read(struct ane_nn *nn, int proc, int i, void *buf, size_t len)
{
	int k = pick(nn, proc, 1, i);

	if (k < 0)
		return k;
	if (len > nn->proc[proc].bo[k].size)
		return -EINVAL;
	memcpy(buf, nn->proc[proc].bo[k].map, len);
	return 0;
}

static int64_t deadline(int timeout_ms)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return (t.tv_sec * 1000000000LL) + t.tv_nsec + (timeout_ms * 1000000LL);
}

/* a signalled fence's error, from its sync file: 0, or a negative errno (-EIO for anything else) */
static int fence_error(struct ane *dev, uint32_t syncobj)
{
	struct drm_syncobj_handle h = {
		.handle = syncobj,
		.flags = DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE,
		.fd = -1,
	};
	struct sync_file_info si = { 0 };
	int r;

	r = io(dev->fd, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &h);
	if (r)
		return r;
	r = io(h.fd, SYNC_IOC_FILE_INFO, &si);
	close(h.fd);
	if (r)
		return r;
	if (si.status == 1)
		return 0;
	return si.status < 0 && si.status > -4096 ? si.status : -EIO;
}

/*
 * wait for the procedure's run until DEADLINE (CLOCK_MONOTONIC ns); the syncobj is cleared
 * unless the wait timed out
 */
static int wait_run(struct ane *dev, struct proc *p, int64_t deadline)
{
	struct drm_syncobj_wait w = {
		.handles = (uintptr_t)&p->sync,
		.count_handles = 1,
		.timeout_nsec = deadline,
	};
	struct drm_syncobj_array a = { .handles = (uintptr_t)&p->sync, .count_handles = 1 };
	int r;

	r = io(dev->fd, DRM_IOCTL_SYNCOBJ_WAIT, &w);
	p->busy = r == -ETIME;
	if (p->busy)
		return r;
	if (!r)
		r = fence_error(dev, p->sync);
	io(dev->fd, DRM_IOCTL_SYNCOBJ_RESET, &a);
	return r;
}

int ane_run(struct ane_nn *nn, int proc, int timeout_ms)
{
	struct ane *dev = nn->dev;
	struct drm_apple_ane_submit s = { 0 };
	struct proc *p;
	int64_t end;
	int r;

	r = prepare(nn, proc);
	if (r)
		return r;
	p = &nn->proc[proc];
	end = deadline(timeout_ms > 0 ? timeout_ms : DEFAULT_TIMEOUT_MS);
	if (p->busy) {
		r = wait_run(dev, p, end);
		if (r == -ETIME)
			return r;
	}
	/* nothing of an earlier run shows through what the engine writes */
	for (unsigned i = 0; i < p->n; i++)
		if (p->info[i].kind != DRM_APPLE_ANE_BUF_INPUT)
			memset(p->bo[i].map, 0, p->bo[i].size);
	s.program = nn->program;
	s.procedure = proc;
	s.buf_count = p->n;
	s.bufs = (uintptr_t)p->sb;
	s.out_sync = p->sync;
	r = io(dev->fd, DRM_IOCTL_APPLE_ANE_SUBMIT, &s);
	if (r)
		return r;
	return wait_run(dev, p, end);
}
