# sparsemat

simple randomized matrix sparsification scheme

## Motivation

(background) Sparsification is a popular method to reduce the time/space costs of (a restricted subset of) linear operations. SVD-based methods optimally represent the largest eigentransformations. Other solutions sacrifice performance on the largest eigenvectors to highlight other details.

This library implements 0th and 1st order derivatives for a particular sparsification represented by products of random sparse matrices. The average length of a minimal path in a random graph being logarithmic suggests the idea would be able to represent many of the unique information routes a fuller matrix can also represent.

In particular, we use stable hashes (rather than explicitly storing indices) to represent the index matrix in a CSR format. Duplicated column indices are correctly handled. The publically exposed type can be trivially mmapped and otherwise treated as a value type. You probably want to use a high sparsity (2-8 columns per row) and a number of factors proportional to `log(row_count) / log(nonzero_columns_per_row)`.

## Installation

Zig has a package manager!!! Do something like the following.

```zig
// build.zig.zon
.{
    .name = "foo",
    .version = "0.0.0",

    .dependencies = .{
        .sparsemat = .{
	   .url = "https://github.com/hmusgrave/sparsemat/archive/refs/tags/0.2.0.tar.gz",
	   .hash = "1220f3264745576c0f46c1a60d64bc9e0aa4f12b08a4cc9771b3d1de2e2e76a42387",
        },
    },
}
```

```zig
// build.zig
const sparsemat_pkg = b.dependency("sparsemat", .{
    .target = target,
    .optimize = optimize,
});
const sparsemat_mod = sparsemat_pkg.module("sparsemat");
exe.addModule("sparsemat", sparsemat_mod);
unit_tests.addModule("sparsemat", sparsemat_mod);
```

## Examples

In general you should read the code to see what works. We expose one type `LogSquareMat` which represents a product of square sparse matrices, and it has methods computing forward and gradient passes.

```zig
const std = @import("std");
const sparsemat = @import("sparsemat");

test {
    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    const L = sparsemat.LogSquareMat(f32, 4, 2, 2);

    // feel free to use an allocator and place it on the heap instead
    var mat: L = undefined;

    // fill the matrix such that its outputs have
    // a standard normal distribution, and use the
    // random seed 314 to define the random sparse
    // connections inside the matrix
    mat.rand_init(rand, 314);

    // set up the problem
    //
    // ordinarily you'd do something like sqrt(sum(square(x-y))) to
    // generate the derivative of some scalar error with respect to
    // the outputs of a matrix multiplication, but that seems a bit
    // wordy for this small example, so we'll just create some values
    // of the right datatype
    const err: [4]f32 = .{ 1, 2, -5, -6 };

    // the gradient computation needs some scratch space, and the result
    // dX needs a place to go
    var buf: [8]f32 = undefined;
    var dX: [4]f32 = undefined;

    // the API takes in everything interesting as pointers, including out
    // parameters
    //
    // Note that we're specifying we'll overwrite dX with the
    // gradient, so there's no need to initialize it ahead of time
    mat.mul_left_vec_dX(&err, &dX, &buf, .overwrite);

    // this time we're accumulating (adding) the new gradient to the old
    // value stored in dX
    mat.mul_left_vec_dX(&err, &dX, &buf, .accumulate);
}
```

## Status

The code works in recent Zig versions (those including https://github.com/ziglang/zig/issues/16695). Make sure to run `zig build test` to check if your compiler is compatible. Performance is adequate (10-14 clock cycles per core per vector per weight on my particular x86_64+AVX2 architecture for a forward+error+backward pass).
