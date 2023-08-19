const std = @import("std");
const native_endian = @import("builtin").target.cpu.arch.endian();

// kind of annoying to have to cast literals as in
//     try expectEqual(@as(T, some_expected_literal), actual_value)
inline fn revEqual(actual: anytype, expected: @TypeOf(actual)) !void {
    return std.testing.expectEqual(expected, actual);
}

inline fn revDeepEqual(actual: anytype, expected: @TypeOf(actual)) !void {
    return std.testing.expectEqualDeep(expected, actual);
}

const M64 = std.math.maxInt(u64);

inline fn leBytes(x: u64) [8]u8 {
    return switch (native_endian) {
        .Big => {
            const le = @byteSwap(x);
            std.mem.asBytes(&le).*;
        },
        .Little => std.mem.asBytes(&x).*,
    };
}

inline fn col_idx(mat_key: u64, row_idx: u64, sparse_col_idx: u64, max_col_idx: u64) u64 {
    const dat = leBytes(mat_key) ++ leBytes(row_idx) ++ leBytes(sparse_col_idx);
    const rng: u64 = std.hash.Wyhash.hash(42, dat[0..]);
    return switch (max_col_idx) {
        0 => 0,
        M64 => rng,
        else => blk: {
            var m: u128 = @intCast(max_col_idx + 1);
            m *= @intCast(rng);
            break :blk @intCast(m >> 64);
        },
    };
}

test "col_idx stable over time and accepts extremal inputs" {
    try revEqual(col_idx(0, 0, 0, 0), 0);
    try revEqual(col_idx(1, 0, 0, 0), 0);
    try revEqual(col_idx(0, 1, 0, 0), 0);
    try revEqual(col_idx(1, 1, 0, 0), 0);
    try revEqual(col_idx(0, 0, 1, 0), 0);
    try revEqual(col_idx(1, 0, 1, 0), 0);
    try revEqual(col_idx(0, 1, 1, 0), 0);
    try revEqual(col_idx(1, 1, 1, 0), 0);
    try revEqual(col_idx(0, 0, 0, M64), 7692060657755225254);
    try revEqual(col_idx(1, 0, 0, M64), 15566565754921268911);
    try revEqual(col_idx(0, 1, 0, M64), 4232397684341721299);
    try revEqual(col_idx(1, 1, 0, M64), 4430779875065412413);
    try revEqual(col_idx(0, 0, 1, M64), 4824904835357622216);
    try revEqual(col_idx(1, 0, 1, M64), 1592624967834318170);
    try revEqual(col_idx(0, 1, 1, M64), 17162690089488699702);
    try revEqual(col_idx(1, 1, 1, M64), 17296532827966587182);
    try revEqual(col_idx(M64, M64, M64, M64), 9161040114273609169);
    try revEqual(col_idx(1, 2, 3, 4), 3);
    try revEqual(col_idx(1, 2, 3, 42), 26);
    try revEqual(col_idx(34, 56, 78, 1), 0);
    try revEqual(col_idx(34, 56, 78, 12345), 3552);
}

test "col_idx max_col_idx doesn't have off-by-one" {
    inline for (0..10) |max_col_idx| {
        var seen_indices = [_]bool{false} ** (max_col_idx + 1);
        for (0..10) |mat_key| {
            for (0..10) |row_idx| {
                for (0..10) |sparse_col_idx| {
                    const idx = col_idx(mat_key, row_idx, sparse_col_idx, max_col_idx);
                    if (idx >= seen_indices.len)
                        return error.ExceededMaxColIdx;
                    seen_indices[idx] = true;
                }
            }
        }

        // strictly speaking, probabilistically we could miss one of the possible
        // return values here, and the test would still be correct
        //
        // however, the implementation is supposed to be stable over time, and
        // this test was both unlikely to fail and did in fact pass when first
        // written, so it shouldn't be flaky going forward
        const seen_indices_vec: @Vector(seen_indices.len, bool) = seen_indices;
        if (!@reduce(.And, seen_indices_vec))
            return error.MissingIdx;
    }
}

fn SparseMat(
    comptime F: type,
    comptime n_row: u64,
    comptime n_col: u64,
    comptime n_nonzero_per_row: u64,
) type {
    return extern struct {
        data: [n_row * n_nonzero_per_row]F,
        mat_idx: u64 = 0,

        pub fn mul_right_vec(
            self: *const @This(),
            in: *const [n_col]F,
            out: *[n_row]F,
        ) void {
            for (out) |*el|
                el.* = 0;
            for (0..n_row) |i_row| {
                for (0..n_nonzero_per_row) |i_sparse_col| {
                    const i_data = i_row * n_nonzero_per_row + i_sparse_col;
                    const i_col = col_idx(self.mat_idx, i_row, i_sparse_col, n_col - 1);
                    out[i_row] = @mulAdd(F, self.data[i_data], in[i_col], out[i_row]);
                }
            }
        }

        pub fn mul_left_vec(
            self: *const @This(),
            in: *const [n_row]F,
            out: *[n_col]F,
        ) void {
            for (out) |*el|
                el.* = 0;
            for (0..n_row) |i_row| {
                for (0..n_nonzero_per_row) |i_sparse_col| {
                    const i_data = i_row * n_nonzero_per_row + i_sparse_col;
                    const i_col = col_idx(self.mat_idx, i_row, i_sparse_col, n_col - 1);
                    out[i_col] = @mulAdd(F, self.data[i_data], in[i_row], out[i_col]);
                }
            }
        }

        pub inline fn mul_right_vec_dX(
            self: *const @This(),
            err: *const [n_row]F,
            out: *[n_col]F,
        ) void {
            return self.mul_left_vec(err, out);
        }

        pub inline fn mul_left_vec_dX(
            self: *const @This(),
            err: *const [n_col]F,
            out: *[n_row]F,
        ) void {
            return self.mul_right_vec(err, out);
        }

        pub fn mul_right_vec_dM(
            self: *const @This(),
            err: *const [n_row]F,
            x: *const [n_col]F,
            out: *@This(),
        ) void {
            for (0..n_row) |i_row| {
                for (0..n_nonzero_per_row) |i_sparse_col| {
                    const i_data = i_row * n_nonzero_per_row + i_sparse_col;
                    const i_col = col_idx(self.mat_idx, i_row, i_sparse_col, n_col - 1);
                    out.data[i_data] = x[i_col] * err[i_row];
                }
            }
            out.mat_idx = self.mat_idx;
        }

        pub fn mul_left_vec_dM(
            self: *const @This(),
            err: *const [n_col]F,
            x: *const [n_row]F,
            out: *@This(),
        ) void {
            for (0..n_row) |i_row| {
                for (0..n_nonzero_per_row) |i_sparse_col| {
                    const i_data = i_row * n_nonzero_per_row + i_sparse_col;
                    const i_col = col_idx(self.mat_idx, i_row, i_sparse_col, n_col - 1);
                    out.data[i_data] = x[i_row] * err[i_col];
                }
            }
            out.mat_idx = self.mat_idx;
        }
    };
}

test "SparseMat compiles and has appropriately matching dimensions" {
    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    const M = SparseMat(f32, 2, 3, 10);
    var mat: M = undefined;
    for (&mat.data) |*el|
        el.* = rand.float(f32) * 2 - 1;
    mat.mat_idx = 0;

    const x: [2]f32 = .{ 3, 4 };
    const y: [3]f32 = .{ -1, 0, 2 };

    var y_out: [3]f32 = undefined;
    var x_out: [2]f32 = undefined;
    var grad_out: M = undefined;

    mat.mul_right_vec(&y, &x_out);
    mat.mul_left_vec(&x, &y_out);

    mat.mul_right_vec_dX(&x, &y_out);
    mat.mul_left_vec_dX(&y, &x_out);

    mat.mul_right_vec_dM(&x, &y, &grad_out);
    mat.mul_left_vec_dM(&y, &x, &grad_out);
}

test "SparseMat well-defined memory layout" {
    const M = SparseMat(f16, 3, 5, 7);
    const z: M = undefined;
    const bytes: [@sizeOf(M)]u8 = @bitCast(z);
    _ = bytes;
}

pub fn LogSquareMat(
    comptime F: type,
    comptime n_dim: u64,
    comptime n_nonzero_per_row: u64,
    comptime n_mat: u64,
) type {
    if (n_mat < 1)
        @compileError("void matrix");
    if (n_dim < 1)
        @compileError("void dimension");
    if (n_nonzero_per_row < 1)
        @compileError("zero matrix");

    const M = SparseMat(F, n_dim, n_dim, n_nonzero_per_row);

    return extern struct {
        mats: [n_mat]M,

        pub fn rand_init(rand: std.rand.Random, mat_idx: u64) @This() {
            var rtn: @This() = undefined;
            rtn.set_root_idx(mat_idx);
            rtn.fill_normal(rand);
            return rtn;
        }

        pub fn set_root_idx(self: *@This(), mat_idx: u64) void {
            for (&self.mats, 0..) |*mat, i|
                mat.mat_idx = mat_idx + i;
        }

        pub fn fill_normal(self: *@This(), rand: std.rand.Random) void {
            for (&self.mats) |*mat| {
                for (&mat.data) |*el| {
                    const f_nzcnt: F = @floatFromInt(n_nonzero_per_row);
                    el.* = rand.floatNorm(F) / @sqrt(f_nzcnt);
                }
            }
        }

        fn debug_indices(self: *const @This()) void {
            for (self.mats, 0..) |mat, mat_i| {
                std.debug.print("Mat ({})\n", .{mat_i});
                for (0..n_dim) |row_i| {
                    for (0..n_nonzero_per_row) |sparse_i| {
                        const col_i = col_idx(mat.mat_idx, row_i, sparse_i, n_dim - 1);
                        std.debug.print("{}, ", .{col_i});
                    }
                    std.debug.print("\n", .{});
                }
                std.debug.print("\n", .{});
            }
        }

        pub fn clone(self: *const @This(), out: *@This()) void {
            for (&self.mats, &out.mats) |*in_mat, *out_mat| {
                for (&in_mat.data, &out_mat.data) |*in_el, *out_el|
                    out_el.* = in_el.*;
                out_mat.mat_idx = in_mat.mat_idx;
            }
        }

        pub fn scale_by(self: *@This(), scalar: F) void {
            for (&self.mats) |*mat| {
                for (&mat.data) |*el|
                    el.* *= scalar;
            }
        }

        pub fn add_from(self: *@This(), other: *const @This()) void {
            for (&self.mats, &other.mats) |*out_mat, *in_mat| {
                for (&out_mat.data, &in_mat.data) |*out_el, *in_el|
                    out_el.* += in_el.*;
            }
        }

        pub inline fn mul_left_vec(
            self: *const @This(),
            x: *const [n_dim]F,
            out: *[n_dim]F,
        ) void {
            var buf: [n_dim * n_mat]F = undefined;
            self.mul_left_vec_for_dM(x, &buf, out);
        }

        pub inline fn mul_right_vec(
            self: *const @This(),
            x: *const [n_dim]F,
            out: *[n_dim]F,
        ) void {
            var buf: [n_dim * n_mat]F = undefined;
            self.mul_right_vec_for_dM(x, &buf, out);
        }

        pub fn mul_left_vec_for_dM(
            self: *const @This(),
            x: *const [n_dim]F,
            out_xs: *[n_dim * n_mat]F,
            out_y: *[n_dim]F,
        ) void {
            out_xs[0..n_dim].* = x.*;
            for (self.mats[0 .. self.mats.len - 1], 0..) |*mat, i| {
                const head = out_xs[i * n_dim ..];
                mat.mul_left_vec(
                    head[0..n_dim],
                    head[n_dim .. 2 * n_dim],
                );
            }
            self.mats[self.mats.len - 1].mul_left_vec(out_xs[(n_mat - 1) * n_dim ..], out_y);
        }

        pub fn mul_right_vec_for_dM(
            self: *const @This(),
            x: *const [n_dim]F,
            out_xs: *[n_dim * n_mat]F,
            out_y: *[n_dim]F,
        ) void {
            out_xs[(n_mat - 1) * n_dim ..].* = x.*;
            var i: usize = n_mat;
            while (i > 1) : (i -= 1) {
                const mat = self.mats[i - 1];
                const head = out_xs[(i - 2) * n_dim ..];
                mat.mul_right_vec(
                    head[n_dim .. 2 * n_dim],
                    head[0..n_dim],
                );
            }
            self.mats[0].mul_right_vec(out_xs[0..n_dim], out_y);
        }

        pub fn mul_left_vec_dX(
            self: *const @This(),
            err: *const [n_dim]F,
            out: *[n_dim]F,
        ) void {
            var err_buf: [n_dim]F = err.*;
            var in_err = &err_buf;
            var out_err = out;
            var i: usize = self.mats.len;
            while (i > 0) : (i -= 1) {
                const mat = self.mats[i - 1];
                mat.mul_left_vec_dX(in_err, out_err);
                std.mem.swap(*[n_dim]F, &in_err, &out_err);
            }
            out.* = in_err.*;
        }

        pub fn mul_right_vec_dX(
            self: *const @This(),
            err: *const [n_dim]F,
            out: *[n_dim]F,
        ) void {
            var err_buf: [n_dim]F = err.*;
            var in_err = &err_buf;
            var out_err = out;
            for (&self.mats) |*mat| {
                mat.mul_right_vec_dX(in_err, out_err);
                std.mem.swap(*[n_dim]F, &in_err, &out_err);
            }
            out.* = in_err.*;
        }

        pub inline fn mul_left_vec_dM(
            self: *const @This(),
            err: *const [n_dim]F,
            xs: *const [n_dim * n_mat]F,
            out: *@This(),
        ) void {
            var buf: [n_dim]F = undefined;
            self.mul_left_vec_dMdX(err, xs, out, &buf);
        }

        pub inline fn mul_right_vec_dM(
            self: *const @This(),
            err: *const [n_dim]F,
            xs: *const [n_dim * n_mat]F,
            out: *@This(),
        ) void {
            var buf: [n_dim]F = undefined;
            self.mul_right_vec_dMdX(err, xs, out, &buf);
        }

        pub fn mul_left_vec_dMdX(
            self: *const @This(),
            err: *const [n_dim]F,
            xs: *const [n_dim * n_mat]F,
            out_dM: *@This(),
            out_dX: *[n_dim]F,
        ) void {
            var err_bufs: [2][n_dim]F = .{ err.*, undefined };
            var in_err: *[n_dim]F = &err_bufs[0];
            var out_err: *[n_dim]F = &err_bufs[1];
            var i: usize = self.mats.len;
            while (i > 0) : (i -= 1) {
                const mat = self.mats[i - 1];
                mat.mul_left_vec_dX(in_err, out_err);
                const head = xs[(i - 1) * n_dim ..];
                mat.mul_left_vec_dM(in_err, head[0..n_dim], &out_dM.mats[i - 1]);
                std.mem.swap(*[n_dim]F, &in_err, &out_err);
            }
            out_dX.* = in_err.*;
        }

        pub fn mul_right_vec_dMdX(
            self: *const @This(),
            err: *const [n_dim]F,
            xs: *const [n_dim * n_mat]F,
            out_dM: *@This(),
            out_dX: *[n_dim]F,
        ) void {
            var err_bufs: [2][n_dim]F = .{ err.*, undefined };
            var in_err: *[n_dim]F = &err_bufs[0];
            var out_err: *[n_dim]F = &err_bufs[1];
            for (&self.mats, &out_dM.mats, 0..) |*mat, *out_mat, i| {
                mat.mul_right_vec_dX(in_err, out_err);
                const head = xs[i * n_dim ..];
                mat.mul_right_vec_dM(in_err, head[0..n_dim], out_mat);
                std.mem.swap(*[n_dim]F, &in_err, &out_err);
            }
            out_dX.* = in_err.*;
        }
    };
}

fn get_test_mat() LogSquareMat(f32, 4, 2, 2) {
    var mat: LogSquareMat(f32, 4, 2, 2) = undefined;
    mat.set_root_idx(42);
    mat.mats[0].data = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    mat.mats[1].data = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    return mat;
}

test "LogSquareMat mul_left_vec" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const x: [4]f32 = .{ 1, 2, 3, 4 };
    var y: [4]f32 = undefined;
    mat.mul_left_vec(&x, &y);
    try revDeepEqual(y, .{ 132, 440, 63, 255 });
}

test "LogSquareMat mul_left_vec_dX" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const err: [4]f32 = .{ 1, 2, 3, 4 };
    var dX: [4]f32 = undefined;
    mat.mul_left_vec_dX(&err, &dX);
    try revDeepEqual(dX, .{ 159, 247, 176, 260 });
}

test "LogSquareMat mul_left_vec_dM" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const x: [4]f32 = .{ 1, 2, 3, 4 };
    const err: [4]f32 = .{ 1, 2, 3, 4 };
    var xs: [8]f32 = undefined;
    var y: [4]f32 = undefined;
    var grad: L = undefined;

    mat.mul_left_vec_for_dM(&x, &xs, &y);
    mat.mul_left_vec_dM(&err, &xs, &grad);

    try revDeepEqual(xs, .{ 1, 2, 3, 4, 28, 33, 40, 9 });
    try revDeepEqual(y, .{ 132, 440, 63, 255 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);
}

test "LogSquareMat mul_left_vec_dMdX" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const x: [4]f32 = .{ 1, 2, 3, 4 };
    const err: [4]f32 = .{ 1, 2, 3, 4 };
    var xs: [8]f32 = undefined;
    var y: [4]f32 = undefined;
    var grad: L = undefined;
    var dX: [4]f32 = undefined;

    mat.mul_left_vec_for_dM(&x, &xs, &y);
    mat.mul_left_vec_dMdX(&err, &xs, &grad, &dX);

    try revDeepEqual(xs, .{ 1, 2, 3, 4, 28, 33, 40, 9 });
    try revDeepEqual(y, .{ 132, 440, 63, 255 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);

    try revDeepEqual(dX, .{ 159, 247, 176, 260 });
}

test "LogSquareMat mul_right_vec" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const x: [4]f32 = .{ 1, 2, 3, 4 };
    var y: [4]f32 = undefined;
    mat.mul_right_vec(&x, &y);
    try revDeepEqual(y, .{ 159, 247, 176, 260 });
}

test "LogSquareMat mul_right_vec_dX" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const err: [4]f32 = .{ 1, 2, 3, 4 };
    var dX: [4]f32 = undefined;
    mat.mul_right_vec_dX(&err, &dX);
    try revDeepEqual(dX, .{ 132, 440, 63, 255 });
}

test "LogSquareMat mul_right_vec_dM" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const x: [4]f32 = .{ 1, 2, 3, 4 };
    const err: [4]f32 = .{ 1, 2, 3, 4 };
    var xs: [8]f32 = undefined;
    var y: [4]f32 = undefined;
    var grad: L = undefined;

    mat.mul_right_vec_for_dM(&x, &xs, &y);
    mat.mul_right_vec_dM(&err, &xs, &grad);

    try revDeepEqual(xs, .{ 12, 16, 22, 53, 1, 2, 3, 4 });
    try revDeepEqual(y, .{ 159, 247, 176, 260 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);
}

test "LogSquareMat mul_right_vec_dMdX" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const x: [4]f32 = .{ 1, 2, 3, 4 };
    const err: [4]f32 = .{ 1, 2, 3, 4 };
    var xs: [8]f32 = undefined;
    var y: [4]f32 = undefined;
    var grad: L = undefined;
    var dX: [4]f32 = undefined;

    mat.mul_right_vec_for_dM(&x, &xs, &y);
    mat.mul_right_vec_dMdX(&err, &xs, &grad, &dX);

    try revDeepEqual(xs, .{ 12, 16, 22, 53, 1, 2, 3, 4 });
    try revDeepEqual(y, .{ 159, 247, 176, 260 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);

    try revDeepEqual(y, .{ 159, 247, 176, 260 });
}

test "LogSquareMat well-defined memory layout" {
    const L = LogSquareMat(f32, 3, 1, 2);
    const z: L = undefined;
    const bytes: [@sizeOf(L)]u8 = @bitCast(z);
    _ = bytes;
}
