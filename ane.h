/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * libane: the Apple Neural Engine through the Linux accel driver apple-ane (today the M3
 * family's, H15).
 *
 * A program is a .hwx, the Mach-O that Apple's compiler builds for H15. It has one or more
 * procedures; a procedure binds numbered buffers: inputs, outputs and intermediates. The library
 * gives each buffer a mapped buffer object the first time a procedure is used, so a run is: write
 * the inputs, ane_run(), read the outputs. The function names follow eiln's libane for the M1's
 * engine; the code is new.
 *
 * Errors: a function that can fail returns a negative errno; one that returns a pointer returns
 * NULL. ane_open() and ane_load*() also set *out to NULL on failure. The usual values:
 *   -ENODEV     no apple-ane node
 *   -EACCES     the node's permissions
 *   -EPERM      loading a program without CAP_SYS_ADMIN where the driver does not confine
 *               user programs
 *   -EINVAL     the driver or the firmware refused the .hwx, or an argument is out of range
 *   -ETIME      a run did not finish within its timeout
 *   -EIO        a run failed on the engine
 *   -ETIMEDOUT  the driver gave up on a run the engine never finished
 *
 * Threads: a struct ane may be used from any thread. A struct ane_nn, and the buffers it maps,
 * are used by one thread at a time; two programs run concurrently, even on one struct ane. Free
 * every program of a device before closing it.
 */
#ifndef ANE_H
#define ANE_H

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ANE_VERSION "0.2.0"

/* an open device: one DRM file, whose programs and buffers no other struct ane sees */
struct ane;
/* a loaded program */
struct ane_nn;

/*
 * Open the device at NODE (/dev/accel/accel0), or with NODE NULL at $ANE_DEVICE, or else the
 * first node under /dev/accel that apple-ane drives.
 */
int ane_open(struct ane **out, const char *node);
void ane_close(struct ane *dev);

/* Load a .hwx from a file or from memory. The bytes are copied: the caller's may go at once. */
int ane_load(struct ane *dev, const char *path, struct ane_nn **out);
int ane_load_mem(struct ane *dev, const void *hwx, size_t size, struct ane_nn **out);
/* Unload a program and unmap its buffers. NULL is ignored. */
void ane_free(struct ane_nn *nn);

/* The program's procedure count. */
int ane_procs(const struct ane_nn *nn);

/*
 * The inputs (src) and outputs (dst) of procedure PROC, each numbered from 0 in the order of their
 * buffer index in the program. The first call for a procedure maps its buffers.
 *   _count  how many
 *   _size   buffer I's bytes (whole pages: the layout inside is the program's, see README.md)
 *   _index  buffer I's index in the program
 *   _ptr    buffer I's mapping, valid until ane_free(); the engine uses it only inside ane_run()
 */
int ane_src_count(struct ane_nn *nn, int proc);
int ane_dst_count(struct ane_nn *nn, int proc);
ssize_t ane_src_size(struct ane_nn *nn, int proc, int i);
ssize_t ane_dst_size(struct ane_nn *nn, int proc, int i);
int ane_src_index(struct ane_nn *nn, int proc, int i);
int ane_dst_index(struct ane_nn *nn, int proc, int i);
void *ane_src_ptr(struct ane_nn *nn, int proc, int i);
const void *ane_dst_ptr(struct ane_nn *nn, int proc, int i);

/* Copy LEN bytes, at most the buffer's size, into input I or out of output I. */
int ane_send(struct ane_nn *nn, int proc, int i, const void *buf, size_t len);
int ane_read(struct ane_nn *nn, int proc, int i, void *buf, size_t len);

/*
 * Run procedure PROC once and wait for it, up to TIMEOUT_MS (3000 when 0 or less). The outputs
 * and intermediates are zeroed first. After -ETIME the run may still be going: the next ane_run()
 * of that procedure waits for it before it starts, and ane_free() leaves its buffers to the
 * driver until it ends.
 */
int ane_run(struct ane_nn *nn, int proc, int timeout_ms);

#ifdef __cplusplus
}
#endif
#endif
