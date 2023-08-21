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
	   .url = "https://github.com/hmusgrave/sparsemat/archive/refs/tags/0.0.1.tar.gz",
            .hash = "1220f41fec4b4a772cb8af612d6a66a8d315240b2c2d9ab6a4eff5e4af88f2e258d0",
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

## Status

The code works in recent Zig versions (those including https://github.com/ziglang/zig/issues/16695). Make sure to run `zig build test` to check if your compiler is compatible. Performance is adequate (10-14 floating point operations per core per vector per weight on my particular x86_64+AVX2 architecture for a forward+error+backward pass, though not very pipelined or optimized).
