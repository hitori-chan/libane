// SPDX-License-Identifier: LGPL-2.1-or-later
//
// anehwx: a CoreML model compiled for the Neural Engine as macOS compiles it, for libane. On macOS,
// as an ordinary user:
//   clang -fobjc-arc -O2 -o anehwx anehwx.m -framework Foundation -framework CoreML
//   ./anehwx MODEL.mlmodel ARCH OUT [SECS]
// ARCH is the engine's target, h15g on the M3 Max.
// CoreML compiles the model (the .mlmodelc's model.espresso.net); Espresso's plan dumps its network
// for the engine (net.plist and its weights) into OUT/esp; ANECompiler's ANECCompile builds
// OUT/model.hwx for ARCH.
// OUT/model.json is libane's metadata: the model's inputs and outputs by name, sorted (taken as the
// order of the program's buffer indexes), with their shapes and dtypes from CoreML's model
// description.
// Then the model runs with one fixed input (x[i] = sin(i * 0.01)) on the CPU alone and on the CPU
// and the engine; the input and both outputs go to OUT/in.bin, OUT/out-cpu.bin and OUT/out-ane.bin
// (float32, the model's shape), the reference for libane's outputs.
// With SECS, a steady load follows for measuring power: batches of 200 predictions on the CPU and
// the engine for SECS seconds, one line a second with the calls in it (`load T CALLS`).
// Espresso and ANECompiler are private frameworks in the dyld cache: reached with dlopen and dlsym,
// not linked.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>

typedef void *(*create_context_fn)(uint64_t, uint64_t);
typedef void *(*create_plan_fn)(void *, uint64_t);
typedef int (*add_network_fn)(void *, const char *, uint64_t, uint64_t *);
typedef int (*plan_fn)(void *);
typedef int (*dump_ir_fn)(void *, char **);
typedef int (*ctx_fn)(void *);
typedef int (*compile_fn)(NSDictionary *, NSDictionary *, void (^)(unsigned int, NSDictionary *));

static void *sym(void *lib, const char *name)
{
	void *p = dlsym(lib, name);
	if (!p)
		fprintf(stderr, "anehwx: no %s: %s\n", name, dlerror());
	return p;
}

static int espresso_ir(NSString *net, NSString *dir)
{
	void *lib =
		dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
	if (!lib) {
		fprintf(stderr, "anehwx: Espresso: %s\n", dlerror());
		return -1;
	}
	create_context_fn create_context = sym(lib, "espresso_create_context");
	create_plan_fn create_plan = sym(lib, "espresso_create_plan");
	add_network_fn add_network = sym(lib, "espresso_plan_add_network");
	plan_fn build = sym(lib, "espresso_plan_build");
	dump_ir_fn dump_ir = sym(lib, "espresso_dump_ir");
	plan_fn destroy_plan = sym(lib, "espresso_plan_destroy");
	ctx_fn destroy_context = sym(lib, "espresso_context_destroy");
	if (!create_context || !create_plan || !add_network || !build || !dump_ir ||
	    !destroy_plan || !destroy_context)
		return -1;
	// the ANE engine's context (10008), as the reference tools ask for it
	void *ctx = create_context(10008, 0xffffffff);
	void *plan = create_plan(ctx, 0);
	uint64_t net_id[2] = { 0 };
	int ret = add_network(plan, net.UTF8String, 0x10010, net_id);
	if (!ret)
		ret = build(plan);
	char *path = (char *)[dir stringByAppendingString:@"/"].UTF8String;
	if (!ret)
		ret = dump_ir(plan, &path);
	printf("anehwx: espresso: %d (%s)\n", ret, ret ? "failed" : path);
	destroy_plan(plan);
	destroy_context(ctx);
	return ret;
}

static int ane_compile(NSString *esp, NSString *out, NSString *arch)
{
	void *lib = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
			   RTLD_NOW);
	if (!lib) {
		fprintf(stderr, "anehwx: ANECompiler: %s\n", dlerror());
		return -1;
	}
	compile_fn compile = sym(lib, "ANECCompile");
	if (!compile)
		return -1;
	NSDictionary *opts = @{
		@"InputNetworks" : @[ @{
			@"NetworkPlistName" : @"net.plist",
			@"NetworkPlistPath" : [esp stringByAppendingString:@"/"]
		} ],
		@"OutputFilePath" : [out stringByAppendingString:@"/"],
		@"OutputFileName" : @"model.hwx",
	};
	NSDictionary *flags = @{ @"TargetArchitecture" : arch };
	int ret = compile(opts, flags, ^(unsigned int status, NSDictionary *d) {
	  printf("anehwx: ANECCompile's status %u%s%s\n", status, status ? ": " : "",
		 status ? d.description.UTF8String : "");
	});
	printf("anehwx: ANECCompile(%s): %d\n", arch.UTF8String, ret);
	return ret;
}

// the sidecar: {"format": "ane-meta-1", "inputs": [{"name", "shape", "dtype"}], "outputs": [...]}
static NSString *dtype_name(MLMultiArrayDataType t)
{
	switch (t) {
	case MLMultiArrayDataTypeFloat16:
		return @"float16";
	case MLMultiArrayDataTypeFloat32:
		return @"float32";
	case MLMultiArrayDataTypeDouble:
		return @"float64";
	case MLMultiArrayDataTypeInt32:
		return @"int32";
	default:
		return @"unknown";
	}
}

static NSArray *feature_list(NSDictionary<NSString *, MLFeatureDescription *> *d)
{
	NSMutableArray *a = [NSMutableArray array];
	for (NSString *name in [d.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		MLMultiArrayConstraint *c = d[name].multiArrayConstraint;
		if (!c)
			continue; // an image or other feature: not a plain tensor, left out
		[a addObject:@{
			@"name" : name,
			@"shape" : c.shape,
			@"dtype" : dtype_name(c.dataType)
		}];
	}
	return a;
}

static int write_meta(NSURL *compiled, NSString *path)
{
	MLModelConfiguration *cfg = [MLModelConfiguration new];
	cfg.computeUnits = MLComputeUnitsCPUOnly;
	NSError *err = nil;
	MLModel *m = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
	if (!m) {
		fprintf(stderr, "anehwx: meta: %s\n", err.description.UTF8String);
		return -1;
	}
	NSDictionary *meta = @{
		@"format" : @"ane-meta-1",
		@"inputs" : feature_list(m.modelDescription.inputDescriptionsByName),
		@"outputs" : feature_list(m.modelDescription.outputDescriptionsByName),
	};
	NSData *j = [NSJSONSerialization
		dataWithJSONObject:meta
			   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
			     error:&err];
	if (!j || ![j writeToFile:path atomically:YES]) {
		fprintf(stderr, "anehwx: meta: %s\n", err.description.UTF8String);
		return -1;
	}
	printf("anehwx: wrote %s\n", path.lastPathComponent.UTF8String);
	return 0;
}

static int run(NSURL *compiled, MLComputeUnits units, NSString *outfile, NSString *infile)
{
	MLModelConfiguration *cfg = [MLModelConfiguration new];
	cfg.computeUnits = units;
	NSError *err = nil;
	MLModel *m = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
	if (!m) {
		fprintf(stderr, "anehwx: load: %s\n", err.description.UTF8String);
		return -1;
	}
	NSString *iname = m.modelDescription.inputDescriptionsByName.allKeys.firstObject;
	NSString *oname = m.modelDescription.outputDescriptionsByName.allKeys.firstObject;
	NSArray<NSNumber *> *shape =
		m.modelDescription.inputDescriptionsByName[iname].multiArrayConstraint.shape;
	MLMultiArray *x = [[MLMultiArray alloc] initWithShape:shape
						     dataType:MLMultiArrayDataTypeFloat32
							error:&err];
	for (NSInteger i = 0; i < x.count; i++)
		((float *)x.dataPointer)[i] = sinf(i * 0.01f);
	if (infile)
		[[NSData dataWithBytes:x.dataPointer length:x.count * 4] writeToFile:infile
									  atomically:YES];
	MLDictionaryFeatureProvider *in =
		[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ iname : x } error:&err];
	id<MLFeatureProvider> o = [m predictionFromFeatures:in error:&err];
	if (!o) {
		fprintf(stderr, "anehwx: predict: %s\n", err.description.UTF8String);
		return -1;
	}
	MLMultiArray *y = [o featureValueForName:oname].multiArrayValue;
	NSMutableData *d = [NSMutableData dataWithLength:y.count * 4];
	for (NSInteger i = 0; i < y.count; i++)
		((float *)d.mutableBytes)[i] = [y[i] floatValue];
	[d writeToFile:outfile atomically:YES];
	printf("anehwx: %s: %ld outputs, y[0] %g, y[last] %g\n",
	       outfile.lastPathComponent.UTF8String, (long)y.count, ((float *)d.mutableBytes)[0],
	       ((float *)d.mutableBytes)[y.count - 1]);
	return 0;
}

static int load(NSURL *compiled, double secs)
{
	MLModelConfiguration *cfg = [MLModelConfiguration new];
	cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
	NSError *err = nil;
	MLModel *m = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
	if (!m)
		return -1;
	NSString *iname = m.modelDescription.inputDescriptionsByName.allKeys.firstObject;
	NSArray<NSNumber *> *shape =
		m.modelDescription.inputDescriptionsByName[iname].multiArrayConstraint.shape;
	MLMultiArray *x = [[MLMultiArray alloc] initWithShape:shape
						     dataType:MLMultiArrayDataTypeFloat32
							error:&err];
	for (NSInteger i = 0; i < x.count; i++)
		((float *)x.dataPointer)[i] = sinf(i * 0.01f);
	MLDictionaryFeatureProvider *in =
		[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ iname : x } error:&err];
	NSMutableArray *batch = [NSMutableArray array];
	for (int i = 0; i < 200; i++)
		[batch addObject:in];
	MLArrayBatchProvider *bp =
		[[MLArrayBatchProvider alloc] initWithFeatureProviderArray:batch];
	CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent(), mark = t0;
	long calls = 0, total = 0;
	while (CFAbsoluteTimeGetCurrent() - t0 < secs) {
		@autoreleasepool {
			if (![m predictionsFromBatch:bp error:&err]) {
				fprintf(stderr, "anehwx: load: %s\n", err.description.UTF8String);
				return -1;
			}
		}
		calls += 200;
		CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
		if (now - mark >= 1.0) {
			printf("load %.1f %ld\n", now - t0, (long)(calls / (now - mark)));
			fflush(stdout);
			total += calls;
			calls = 0;
			mark = now;
		}
	}
	total += calls;
	printf("anehwx: load: %ld calls in %.1f s, %.0f a second\n", total,
	       CFAbsoluteTimeGetCurrent() - t0, total / (CFAbsoluteTimeGetCurrent() - t0));
	return 0;
}

int main(int argc, char *argv[])
{
	@autoreleasepool {
		if (argc != 4 && argc != 5) {
			fprintf(stderr, "usage: anehwx MODEL.mlmodel ARCH OUT [SECS]\n");
			return 2;
		}
		NSString *out = [NSString stringWithUTF8String:argv[3]];
		NSString *esp = [out stringByAppendingPathComponent:@"esp"];
		[[NSFileManager defaultManager] createDirectoryAtPath:esp
					  withIntermediateDirectories:YES
							   attributes:nil
								error:nil];
		NSError *err = nil;
		NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:@(argv[1])]
						       error:&err];
		if (!compiled) {
			fprintf(stderr, "anehwx: compile: %s\n", err.description.UTF8String);
			return 1;
		}
		printf("anehwx: compiled to %s\n", compiled.path.UTF8String);
		int r1 = espresso_ir(
			[compiled.path stringByAppendingPathComponent:@"model.espresso.net"], esp);
		int r2 = r1 ? -1 : ane_compile(esp, out, @(argv[2]));
		int r3 = run(compiled, MLComputeUnitsCPUOnly,
			     [out stringByAppendingPathComponent:@"out-cpu.bin"],
			     [out stringByAppendingPathComponent:@"in.bin"]);
		int r4 = run(compiled, MLComputeUnitsCPUAndNeuralEngine,
			     [out stringByAppendingPathComponent:@"out-ane.bin"], nil);
		int r6 = write_meta(compiled, [out stringByAppendingPathComponent:@"model.json"]);
		int r5 = argc == 5 ? load(compiled, atof(argv[4])) : 0;
		return r1 || r2 || r3 || r4 || r5 || r6;
	}
}
