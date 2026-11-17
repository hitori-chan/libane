// SPDX-License-Identifier: LGPL-2.1-or-later
/*
 * libane's tests, in TAP: test [MODEL...], a MODEL being a .hwx or a directory holding model.hwx.
 *
 * Without an apple-ane node only the device-free checks run. With one, each program adds: the
 * API's contracts and errors, reruns that match bit for bit, recovery from a run that times out,
 * load-run-free cycles that leak no descriptor or memory, and runs from several threads at once.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../ane.h"

#define CYCLES	     100
#define THREADS	     4
#define THREAD_RUNS  50
#define TIMEOUT_TRYS 500

#ifdef __SANITIZE_ADDRESS__
#define SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define SANITIZED 1
#endif
#endif
#ifndef SANITIZED
#define SANITIZED 0
#endif

static int tests, failed;

static int ok(int cond, const char *fmt, ...)
{
	va_list ap;

	tests++;
	if (!cond)
		failed++;
	printf("%sok %d - ", cond ? "" : "not ", tests);
	va_start(ap, fmt);
	vprintf(fmt, ap);
	va_end(ap);
	putchar('\n');
	fflush(stdout);
	return cond;
}

static void skip(const char *what, const char *why)
{
	tests++;
	printf("ok %d - %s # SKIP %s\n", tests, what, why);
}

static int count_fds(void)
{
	DIR *d = opendir("/proc/self/fd");
	struct dirent *e;
	int n = 0;

	if (!d)
		return -1;
	while ((e = readdir(d)))
		n += e->d_name[0] != '.';
	closedir(d);
	return n - 1; /* the DIR's own */
}

static long rss_kb(void)
{
	FILE *f = fopen("/proc/self/status", "r");
	char line[256];
	long kb = -1;

	if (!f)
		return -1;
	while (fgets(line, sizeof(line), f))
		if (strncmp(line, "VmRSS:", 6) == 0) {
			kb = strtol(line + 6, NULL, 10);
			break;
		}
	fclose(f);
	return kb;
}

static void *slurp(const char *path, size_t *size)
{
	int fd = open(path, O_RDONLY | O_CLOEXEC);
	unsigned char *buf = NULL;
	struct stat st;
	size_t n = 0;

	*size = 0;
	if (fd < 0)
		return NULL;
	if (fstat(fd, &st) == 0 && st.st_size > 0)
		buf = malloc((size_t)st.st_size);
	while (buf && n < (size_t)st.st_size) {
		ssize_t got = read(fd, buf + n, (size_t)st.st_size - n);

		if (got <= 0) {
			free(buf);
			buf = NULL;
		} else {
			n += (size_t)got;
		}
	}
	close(fd);
	if (buf)
		*size = n;
	return buf;
}

/* inputs from a pattern of float16 values, shifted by SEED */
static void fill(struct ane_nn *nn, int proc, unsigned seed)
{
	static const uint16_t pat[] = { 0x0000, 0x3c00, 0xbc00, 0x3800, 0xb800, 0x3400, 0x4000 };
	int n = ane_src_count(nn, proc);

	for (int i = 0; i < n; i++) {
		uint16_t *h = ane_src_ptr(nn, proc, i);
		size_t m = (size_t)ane_src_size(nn, proc, i) / 2;

		for (size_t k = 0; k < m; k++)
			h[k] = pat[(k + seed) % (sizeof(pat) / sizeof(pat[0]))];
	}
}

/* the outputs' bytes, end to end */
static unsigned char *snap(struct ane_nn *nn, int proc, size_t *len)
{
	int n = ane_dst_count(nn, proc);
	unsigned char *buf;
	size_t total = 0, at = 0;

	for (int i = 0; i < n; i++)
		total += (size_t)ane_dst_size(nn, proc, i);
	buf = malloc(total ? total : 1);
	if (!buf)
		abort();
	for (int i = 0; i < n; i++) {
		size_t s = (size_t)ane_dst_size(nn, proc, i);

		memcpy(buf + at, ane_dst_ptr(nn, proc, i), s);
		at += s;
	}
	*len = total;
	return buf;
}

static int same(const unsigned char *a, size_t na, const unsigned char *b, size_t nb)
{
	return na == nb && memcmp(a, b, na) == 0;
}

/* run procedure 0 on the seed-0 inputs: whether it succeeds with outputs REF */
static int run_matches(struct ane_nn *nn, const unsigned char *ref, size_t nref)
{
	unsigned char *y;
	size_t ny;
	int r;

	fill(nn, 0, 0);
	r = ane_run(nn, 0, 0);
	if (r)
		return 0;
	y = snap(nn, 0, &ny);
	r = same(y, ny, ref, nref);
	free(y);
	return r;
}

static void device_free(void)
{
	struct ane *dev = (struct ane *)1;

	ok(ane_open(&dev, "/nonexistent/accel0") == -ENOENT && !dev,
	   "ane_open: a missing node is -ENOENT");
	dev = (struct ane *)1;
	ok(ane_open(&dev, "/dev/null") == -ENODEV && !dev,
	   "ane_open: a node not apple-ane's is -ENODEV");
	ane_free(NULL);
	ane_close(NULL);
	ok(1, "ane_free(NULL) and ane_close(NULL) are no-ops");
}

static void load_errors(struct ane *dev)
{
	char junk[4096], tmp[] = "/tmp/libane-test-XXXXXX";
	struct ane_nn *nn = (struct ane_nn *)1;
	int fd, r;

	ok(ane_load(dev, "/nonexistent.hwx", &nn) == -ENOENT && !nn,
	   "ane_load: a missing file is -ENOENT");
	nn = (struct ane_nn *)1;
	ok(ane_load(dev, "/", &nn) == -EINVAL && !nn, "ane_load: a directory is -EINVAL");
	fd = mkstemp(tmp);
	if (fd >= 0) {
		close(fd);
		nn = (struct ane_nn *)1;
		ok(ane_load(dev, tmp, &nn) == -EINVAL && !nn, "ane_load: an empty file is -EINVAL");
		unlink(tmp);
	}
	nn = (struct ane_nn *)1;
	ok(ane_load_mem(dev, NULL, 16, &nn) == -EINVAL && !nn, "ane_load_mem: NULL is -EINVAL");
	nn = (struct ane_nn *)1;
	ok(ane_load_mem(dev, junk, 0, &nn) == -EINVAL && !nn, "ane_load_mem: 0 bytes is -EINVAL");
	memset(junk, 0xa5, sizeof(junk));
	nn = (struct ane_nn *)1;
	r = ane_load_mem(dev, junk, sizeof(junk), &nn);
	ok(r == -EINVAL && !nn, "ane_load_mem: 4 KiB of junk is -EINVAL (%d)", r);
}

struct job {
	struct ane *dev; /* NULL: the thread opens its own */
	const void *hwx;
	size_t size;
	const unsigned char *ref;
	size_t nref;
	int bad;
};

static void *worker(void *arg)
{
	struct job *j = arg;
	struct ane *dev = j->dev;
	struct ane_nn *nn;

	if (!dev && ane_open(&dev, NULL)) {
		j->bad = THREAD_RUNS;
		return NULL;
	}
	if (ane_load_mem(dev, j->hwx, j->size, &nn)) {
		j->bad = THREAD_RUNS;
	} else {
		for (int k = 0; k < THREAD_RUNS; k++)
			j->bad += !run_matches(nn, j->ref, j->nref);
		ane_free(nn);
	}
	if (!j->dev)
		ane_close(dev);
	return NULL;
}

static int threads(struct ane *shared, const void *hwx, size_t size, const unsigned char *ref,
		   size_t nref, int n)
{
	pthread_t t[THREADS];
	struct job j[THREADS];
	int bad = 0;

	for (int i = 0; i < n; i++) {
		j[i] = (struct job){ shared, hwx, size, ref, nref, 0 };
		if (pthread_create(&t[i], NULL, worker, &j[i]))
			abort();
	}
	for (int i = 0; i < n; i++) {
		pthread_join(t[i], NULL);
		bad += j[i].bad;
	}
	return bad;
}

static void contracts(struct ane_nn *nn, const char *name, int procs)
{
	long page = sysconf(_SC_PAGESIZE);
	char buf[8];

	for (int p = 0; p < procs; p++) {
		int ns = ane_src_count(nn, p), nd = ane_dst_count(nn, p), good = ns >= 0 && nd >= 1;
		int last = -1;

		for (int i = 0; good && i < ns; i++) {
			ssize_t s = ane_src_size(nn, p, i);
			int x = ane_src_index(nn, p, i);

			good = s > 0 && s % page == 0 && x > last && ane_src_ptr(nn, p, i);
			last = x;
		}
		last = -1;
		for (int i = 0; good && i < nd; i++) {
			ssize_t s = ane_dst_size(nn, p, i);
			int x = ane_dst_index(nn, p, i);

			good = s > 0 && s % page == 0 && x > last && ane_dst_ptr(nn, p, i);
			last = x;
		}
		ok(good, "%s: procedure %d: %d inputs, %d outputs, mapped, whole pages, by index",
		   name, p, ns, nd);
		if (ns > 0) {
			ssize_t s = ane_src_size(nn, p, 0);
			unsigned char *big = calloc(1, (size_t)s + 1);

			if (!big)
				abort();
			ok(ane_send(nn, p, 0, big, (size_t)s + 1) == -EINVAL &&
				   ane_read(nn, p, 0, big, (size_t)ane_dst_size(nn, p, 0) + 1) ==
					   -EINVAL,
			   "%s: ane_send and ane_read past a buffer's end are -EINVAL", name);
			memset(big, 0x5a, (size_t)s);
			ok(ane_send(nn, p, 0, big, (size_t)s) == 0 &&
				   memcmp(ane_src_ptr(nn, p, 0), big, (size_t)s) == 0,
			   "%s: ane_send writes the mapping", name);
			free(big);
		}
	}
	ok(ane_src_count(nn, -1) == -EINVAL && ane_dst_count(nn, procs) == -EINVAL &&
		   ane_run(nn, procs, 0) == -EINVAL,
	   "%s: a procedure out of range is -EINVAL", name);
	ok(ane_src_size(nn, 0, -1) == -EINVAL &&
		   ane_dst_index(nn, 0, ane_dst_count(nn, 0)) == -EINVAL &&
		   !ane_src_ptr(nn, 0, ane_src_count(nn, 0)) && !ane_dst_ptr(nn, 0, -1) &&
		   ane_read(nn, 0, ane_dst_count(nn, 0), buf, 1) == -EINVAL,
	   "%s: a buffer out of range is -EINVAL or NULL", name);
}

static void model(struct ane *dev, const char *arg)
{
	char path[4096];
	const char *name = arg;
	struct stat st;
	struct ane_nn *nn, *nn2;
	unsigned char *hwx, *a, *b;
	size_t size, na, nb;
	int procs, r, fd0, hit = 0;
	long rss0, rss1;

	if (stat(arg, &st) == 0 && S_ISDIR(st.st_mode))
		snprintf(path, sizeof(path), "%s/model.hwx", arg);
	else
		snprintf(path, sizeof(path), "%s", arg);
	hwx = slurp(path, &size);
	ok(hwx != NULL, "%s: read %s", name, path);
	if (!hwx)
		return;
	r = ane_load_mem(dev, hwx, size / 2, &nn);
	ok(r < 0 && !nn, "%s: the first half of the .hwx is refused (%d)", name, r);
	r = ane_load_mem(dev, hwx, 64, &nn);
	ok(r < 0 && !nn, "%s: its first 64 bytes are refused (%d)", name, r);
	r = ane_load(dev, path, &nn);
	if (!ok(r == 0, "%s: ane_load (%d)", name, r)) {
		free(hwx);
		return;
	}
	procs = ane_procs(nn);
	ok(procs >= 1, "%s: %d procedures", name, procs);
	contracts(nn, name, procs);

	/* procedure 0's outputs on the seed-0 inputs are the reference from here on */
	fill(nn, 0, 0);
	r = ane_run(nn, 0, 0);
	ok(r == 0, "%s: a run (%d)", name, r);
	a = snap(nn, 0, &na);
	for (int i = 0; i < ane_dst_count(nn, 0); i++) {
		void *o = (void *)ane_dst_ptr(nn, 0, i);

		if (o)
			memset(o, 0xff, (size_t)ane_dst_size(nn, 0, i));
	}
	fill(nn, 0, 0);
	r = ane_run(nn, 0, 0);
	b = snap(nn, 0, &nb);
	ok(r == 0 && same(a, na, b, nb), "%s: a rerun over scribbled outputs matches bit for bit",
	   name);
	free(b);
	fill(nn, 0, 3);
	r = ane_run(nn, 0, 0);
	b = snap(nn, 0, &nb);
	ok(r == 0 && !same(a, na, b, nb), "%s: other inputs give other outputs", name);
	free(b);
	if (ane_dst_count(nn, 0) > 0) {
		size_t s = (size_t)ane_dst_size(nn, 0, 0);
		unsigned char *c = malloc(s);

		ok(c && ane_read(nn, 0, 0, c, s) == 0 && memcmp(c, ane_dst_ptr(nn, 0, 0), s) == 0,
		   "%s: ane_read copies the mapping", name);
		free(c);
	}
	for (int p = 1; p < procs; p++) {
		fill(nn, p, 0);
		r = ane_run(nn, p, 0);
		ok(r == 0, "%s: procedure %d runs (%d)", name, p, r);
	}
	r = ane_load_mem(dev, hwx, size, &nn2);
	ok(r == 0 && run_matches(nn2, a, na), "%s: ane_load_mem's program runs the same (%d)", name,
	   r);
	if (!r)
		ane_free(nn2);

	/* a 1 ms timeout until a run outlasts it, then a rerun that must wait for it */
	for (int t = 0; t < TIMEOUT_TRYS && !hit; t++) {
		fill(nn, 0, 0);
		r = ane_run(nn, 0, 1);
		if (r == -ETIME)
			hit = 1;
		else if (r)
			break;
	}
	if (hit) {
		ok(run_matches(nn, a, na), "%s: after -ETIME, the next run waits and matches",
		   name);
		r = ane_load_mem(dev, hwx, size, &nn2);
		hit = 0;
		for (int t = 0; !r && t < TIMEOUT_TRYS && !hit; t++) {
			fill(nn2, 0, 0);
			hit = ane_run(nn2, 0, 1) == -ETIME;
		}
		if (!r)
			ane_free(nn2);
		ok(!r, "%s: ane_free() with a run still going (%s)", name,
		   hit ? "it timed out" : "none timed out");
	} else if (r) {
		ok(0, "%s: a 1 ms run failed (%d)", name, r);
	} else {
		skip("after -ETIME, the next run waits and matches", "every run ended within 1 ms");
	}

	/* load-run-free cycles; the first warms up the allocator */
	r = ane_load_mem(dev, hwx, size, &nn2);
	if (!r)
		ane_free(nn2);
	fd0 = count_fds();
	rss0 = rss_kb();
	r = 0;
	for (int c = 0; c < CYCLES; c++) {
		if (ane_load_mem(dev, hwx, size, &nn2)) {
			r++;
			continue;
		}
		r += !run_matches(nn2, a, na);
		ane_free(nn2);
	}
	rss1 = rss_kb();
	ok(r == 0, "%s: %d load-run-free cycles match", name, CYCLES);
	ok(count_fds() == fd0, "%s: no descriptor leaks over the cycles (%d)", name, fd0);
	if (SANITIZED)
		skip("memory over the cycles",
		     "ASan's quarantine holds freed memory; LeakSanitizer checks");
	else
		ok(rss1 - rss0 < 2048, "%s: memory over the cycles: %+ld KiB", name, rss1 - rss0);

	r = threads(dev, hwx, size, a, na, THREADS);
	ok(r == 0, "%s: %d threads on one device, %d runs each, all match (%d bad)", name, THREADS,
	   THREAD_RUNS, r);
	r = threads(NULL, hwx, size, a, na, 2);
	ok(r == 0, "%s: 2 threads on devices of their own, all match (%d bad)", name, r);

	ane_free(nn);
	free(a);
	free(hwx);
}

int main(int argc, char **argv)
{
	struct ane *dev;
	int r, fd0;

	device_free();
	r = ane_open(&dev, NULL);
	if (r) {
		printf("# no usable apple-ane node (%s): the device's checks skipped\n",
		       strerror(-r));
		printf("1..%d\n", tests);
		return failed != 0;
	}
	ane_close(dev);
	setenv("ANE_DEVICE", "/dev/null", 1);
	ok(ane_open(&dev, NULL) == -ENODEV, "ane_open: $ANE_DEVICE is used");
	unsetenv("ANE_DEVICE");
	fd0 = count_fds();
	for (int i = 0; i < CYCLES; i++) {
		if (ane_open(&dev, NULL) == 0)
			ane_close(dev);
	}
	ok(count_fds() == fd0, "ane_open and ane_close %d times leak no descriptor", CYCLES);
	r = ane_open(&dev, NULL);
	if (!ok(r == 0, "ane_open (%d)", r))
		return 1;
	load_errors(dev);
	for (int i = 1; i < argc; i++)
		model(dev, argv[i]);
	if (argc < 2)
		skip("the program checks", "no MODEL given");
	ane_close(dev);
	printf("1..%d\n", tests);
	if (failed)
		printf("# %d of %d failed\n", failed, tests);
	return failed != 0;
}
