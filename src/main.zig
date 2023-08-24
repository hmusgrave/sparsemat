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

// in: 0
// out: 1
// bufa: 2
// bufb: 3
const pingpong = struct {
    next_round: usize = 0,

    pub inline fn next(self: *@This(), is_final: bool) [2]usize {
        const out_round = if (is_final) 1 else 2 + (self.next_round & 1);
        const in_round = if (self.next_round == 0) 0 else 3 - (self.next_round & 1);
        self.next_round += 1;
        return .{ in_round, out_round };
    }
};

test "pingpong" {
    var data: [4]usize = undefined;

    for (1..10) |n_rounds| {
        data[0] = 0;
        var tracker = pingpong{};
        for (0..n_rounds) |round| {
            const inout = tracker.next(round == n_rounds - 1);
            const in = inout[0];
            const out = inout[1];
            data[out] = data[in] + 1;
        }
        try revEqual(data[1], n_rounds);
    }
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

const OutParamStrategy = enum {
    overwrite,
    accumulate,
};

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
            comptime strat: OutParamStrategy,
        ) void {
            switch (strat) {
                .overwrite => {
                    for (out) |*el|
                        el.* = 0;
                },
                .accumulate => {},
            }
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
            comptime strat: OutParamStrategy,
        ) void {
            switch (strat) {
                .overwrite => {
                    for (out) |*el|
                        el.* = 0;
                },
                .accumulate => {},
            }
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
            comptime strat: OutParamStrategy,
        ) void {
            self.mul_left_vec(err, out, strat);
        }

        pub inline fn mul_left_vec_dX(
            self: *const @This(),
            err: *const [n_col]F,
            out: *[n_row]F,
            comptime strat: OutParamStrategy,
        ) void {
            self.mul_right_vec(err, out, strat);
        }

        pub fn mul_right_vec_dM(
            self: *const @This(),
            err: *const [n_row]F,
            x: *const [n_col]F,
            out: *@This(),
            comptime strat: OutParamStrategy,
        ) !void {
            switch (strat) {
                .overwrite => {
                    out.mat_idx = self.mat_idx;
                },
                .accumulate => {
                    if (out.mat_idx != self.mat_idx)
                        return error.MismatchedStructure;
                },
            }
            for (0..n_row) |i_row| {
                for (0..n_nonzero_per_row) |i_sparse_col| {
                    const i_data = i_row * n_nonzero_per_row + i_sparse_col;
                    const i_col = col_idx(self.mat_idx, i_row, i_sparse_col, n_col - 1);
                    switch (strat) {
                        .overwrite => {
                            out.data[i_data] = x[i_col] * err[i_row];
                        },
                        .accumulate => {
                            out.data[i_data] = @mulAdd(F, x[i_col], err[i_row], out.data[i_data]);
                        },
                    }
                }
            }
        }

        pub fn mul_left_vec_dM(
            self: *const @This(),
            err: *const [n_col]F,
            x: *const [n_row]F,
            out: *@This(),
            comptime strat: OutParamStrategy,
        ) !void {
            switch (strat) {
                .overwrite => {
                    out.mat_idx = self.mat_idx;
                },
                .accumulate => {
                    if (out.mat_idx != self.mat_idx)
                        return error.MismatchedStructure;
                },
            }
            for (0..n_row) |i_row| {
                for (0..n_nonzero_per_row) |i_sparse_col| {
                    const i_data = i_row * n_nonzero_per_row + i_sparse_col;
                    const i_col = col_idx(self.mat_idx, i_row, i_sparse_col, n_col - 1);
                    switch (strat) {
                        .overwrite => {
                            out.data[i_data] = x[i_row] * err[i_col];
                        },
                        .accumulate => {
                            out.data[i_data] = @mulAdd(F, x[i_row], err[i_col], out.data[i_data]);
                        },
                    }
                }
            }
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

    mat.mul_right_vec(&y, &x_out, .overwrite);
    mat.mul_left_vec(&x, &y_out, .overwrite);

    mat.mul_right_vec_dX(&x, &y_out, .overwrite);
    mat.mul_left_vec_dX(&y, &x_out, .overwrite);

    try mat.mul_right_vec_dM(&x, &y, &grad_out, .overwrite);
    try mat.mul_left_vec_dM(&y, &x, &grad_out, .overwrite);
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

        pub fn rand_init(self: *@This(), rand: std.rand.Random, mat_idx: u64) void {
            self.set_root_idx(mat_idx);
            self.fill_normal(rand);
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

        pub fn clone_to(self: *const @This(), out: *@This()) void {
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

        pub fn mul_left_vec(
            self: *const @This(),
            x: *const [n_dim]F,
            out: *[n_dim]F,
            buf: *[2 * n_dim]F,
        ) void {
            if (n_mat == 1) {
                self.mats[0].mul_left_vec(x, out);
                return;
            }

            var tracker = pingpong{};
            const ptrs = [_]*[n_dim]F{ undefined, out, buf[0..n_dim], buf[n_dim..][0..n_dim] };
            self.mats[0].mul_left_vec(x, ptrs[tracker.next(false)[1]], .overwrite);

            for (self.mats[1..], 1..n_mat) |*mat, round| {
                const inout = tracker.next(round == n_mat - 1);
                mat.mul_left_vec(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
            }
        }

        pub fn mul_right_vec(
            self: *const @This(),
            x: *const [n_dim]F,
            out: *[n_dim]F,
            buf: *[2 * n_dim]F,
        ) void {
            if (n_mat == 1) {
                self.mats[0].mul_right_vec(x, out);
                return;
            }

            var tracker = pingpong{};
            const ptrs = [_]*[n_dim]F{ undefined, out, buf[0..n_dim], buf[n_dim..][0..n_dim] };
            self.mats[n_mat - 1].mul_right_vec(x, ptrs[tracker.next(false)[1]], .overwrite);

            for (1..n_mat) |round| {
                const mat = &self.mats[n_mat - round - 1];
                const inout = tracker.next(round == n_mat - 1);
                mat.mul_right_vec(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
            }
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
                    .overwrite,
                );
            }
            self.mats[self.mats.len - 1].mul_left_vec(out_xs[(n_mat - 1) * n_dim ..], out_y, .overwrite);
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
                const mat = &self.mats[i - 1];
                const head = out_xs[(i - 2) * n_dim ..];
                mat.mul_right_vec(
                    head[n_dim .. 2 * n_dim],
                    head[0..n_dim],
                    .overwrite,
                );
            }
            self.mats[0].mul_right_vec(out_xs[0..n_dim], out_y, .overwrite);
        }

        pub fn mul_left_vec_dX(
            self: *const @This(),
            err: *const [n_dim]F,
            out: *[n_dim]F,
            buf: *[2 * n_dim]F,
            comptime strat: OutParamStrategy,
        ) void {
            if (n_mat == 1) {
                self.mats[0].mul_left_vec_dX(err, out, strat);
                return;
            }

            var tracker = pingpong{};
            const ptrs = [_]*[n_dim]F{ undefined, out, buf[0..n_dim], buf[n_dim..][0..n_dim] };
            self.mats[n_mat - 1].mul_left_vec_dX(err, ptrs[tracker.next(false)[1]], .overwrite);

            for (1..n_mat - 1) |round| {
                const mat = &self.mats[n_mat - round - 1];
                const inout = tracker.next(false);
                mat.mul_left_vec_dX(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
            }

            {
                const mat = &self.mats[n_mat - (n_mat - 1) - 1];
                const inout = tracker.next(true);
                mat.mul_left_vec_dX(ptrs[inout[0]], ptrs[inout[1]], strat);
            }
        }

        pub fn mul_right_vec_dX(
            self: *const @This(),
            err: *const [n_dim]F,
            out: *[n_dim]F,
            buf: *[2 * n_dim]F,
            comptime strat: OutParamStrategy,
        ) void {
            if (n_mat == 1) {
                self.mats[0].mul_right_vec_dX(err, out, strat);
                return;
            }

            var tracker = pingpong{};
            const ptrs = [_]*[n_dim]F{ undefined, out, buf[0..n_dim], buf[n_dim..][0..n_dim] };
            self.mats[0].mul_right_vec_dX(err, ptrs[tracker.next(false)[1]], .overwrite);

            for (self.mats[1 .. n_mat - 1]) |*mat| {
                const inout = tracker.next(false);
                mat.mul_right_vec_dX(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
            }

            {
                const inout = tracker.next(true);
                const mat = &self.mats[n_mat - 1];
                mat.mul_right_vec_dX(ptrs[inout[0]], ptrs[inout[1]], strat);
            }
        }

        pub fn mul_left_vec_dM(
            self: *const @This(),
            err: *const [n_dim]F,
            xs: *const [n_dim * n_mat]F,
            out: *@This(),
            buf: *[2 * n_dim]F,
            comptime strat: OutParamStrategy,
        ) !void {
            if (n_mat == 1) {
                self.mats[0].mul_left_vec_dM(err, xs[0..n_dim], &out.mats[0], strat);
                return;
            }

            var tracker = pingpong{};
            const ptrs = [_]*[n_dim]F{ undefined, undefined, buf[0..n_dim], buf[n_dim..][0..n_dim] };
            {
                const inout = tracker.next(false);
                self.mats[n_mat - 1].mul_left_vec_dX(err, ptrs[inout[1]], .overwrite);
                try self.mats[n_mat - 1].mul_left_vec_dM(err, xs[(n_mat - 1) * n_dim ..][0..n_dim], &out.mats[n_mat - 1], strat);
            }

            for (1..n_mat - 1) |round| {
                const i = n_mat - round - 1;
                const mat = &self.mats[i];
                const inout = tracker.next(false);
                mat.mul_left_vec_dX(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
                try mat.mul_left_vec_dM(ptrs[inout[0]], xs[i * n_dim ..][0..n_dim], &out.mats[i], strat);
            }

            {
                const mat = &self.mats[0];
                const inout = tracker.next(false);
                mat.mul_left_vec_dX(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
                try mat.mul_left_vec_dM(ptrs[inout[0]], xs[0..n_dim], &out.mats[0], strat);
            }
        }

        pub inline fn mul_right_vec_dM(
            self: *const @This(),
            err: *const [n_dim]F,
            xs: *const [n_dim * n_mat]F,
            out: *@This(),
            buf: *[2 * n_dim]F,
            comptime strat: OutParamStrategy,
        ) !void {
            if (n_mat == 1) {
                self.mats[0].mul_right_vec_dM(err, xs[0..n_dim], &out.mats[0], strat);
                return;
            }

            var tracker = pingpong{};
            const ptrs = [_]*[n_dim]F{ undefined, undefined, buf[0..n_dim], buf[n_dim..][0..n_dim] };
            {
                const inout = tracker.next(false);
                self.mats[0].mul_right_vec_dX(err, ptrs[inout[1]], .overwrite);
                try self.mats[0].mul_right_vec_dM(err, xs[0..n_dim], &out.mats[0], strat);
            }

            for (self.mats[1 .. n_mat - 1], out.mats[1 .. n_mat - 1], 1..) |*mat, *out_mat, i| {
                const inout = tracker.next(false);
                mat.mul_right_vec_dX(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
                try mat.mul_right_vec_dM(ptrs[inout[0]], xs[i * n_dim ..][0..n_dim], out_mat, strat);
            }

            {
                const mat = &self.mats[n_mat - 1];
                var out_mat = &out.mats[n_mat - 1];
                const inout = tracker.next(false);
                mat.mul_right_vec_dX(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
                try mat.mul_right_vec_dM(ptrs[inout[0]], xs[(n_mat - 1) * n_dim ..][0..n_dim], out_mat, strat);
            }
        }

        pub fn mul_left_vec_dMdX(
            self: *const @This(),
            err: *const [n_dim]F,
            xs: *const [n_dim * n_mat]F,
            out_dM: *@This(),
            out_dX: *[n_dim]F,
            buf: *[2 * n_dim]F,
            comptime strat: OutParamStrategy,
        ) !void {
            if (n_mat == 1) {
                self.mats[0].mul_left_vec_dM(err, xs[0..n_dim], &out_dM.mats[0], strat);
                return;
            }

            var tracker = pingpong{};
            const ptrs = [_]*[n_dim]F{ undefined, out_dX, buf[0..n_dim], buf[n_dim..][0..n_dim] };
            {
                const inout = tracker.next(false);
                self.mats[n_mat - 1].mul_left_vec_dX(err, ptrs[inout[1]], .overwrite);
                try self.mats[n_mat - 1].mul_left_vec_dM(err, xs[(n_mat - 1) * n_dim ..][0..n_dim], &out_dM.mats[n_mat - 1], strat);
            }

            for (1..n_mat - 1) |round| {
                const i = n_mat - round - 1;
                const mat = &self.mats[i];
                const inout = tracker.next(false);
                mat.mul_left_vec_dX(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
                try mat.mul_left_vec_dM(ptrs[inout[0]], xs[i * n_dim ..][0..n_dim], &out_dM.mats[i], strat);
            }

            {
                const mat = &self.mats[0];
                const inout = tracker.next(true);
                mat.mul_left_vec_dX(ptrs[inout[0]], ptrs[inout[1]], strat);
                try mat.mul_left_vec_dM(ptrs[inout[0]], xs[0..n_dim], &out_dM.mats[0], strat);
            }
        }

        pub fn mul_right_vec_dMdX(
            self: *const @This(),
            err: *const [n_dim]F,
            xs: *const [n_dim * n_mat]F,
            out_dM: *@This(),
            out_dX: *[n_dim]F,
            buf: *[2 * n_dim]F,
            comptime strat: OutParamStrategy,
        ) !void {
            if (n_mat == 1) {
                try self.mats[0].mul_right_vec_dM(err, xs[0..n_dim], &out_dM.mats[0], strat);
                return;
            }

            var tracker = pingpong{};
            const ptrs = [_]*[n_dim]F{ undefined, out_dX, buf[0..n_dim], buf[n_dim..][0..n_dim] };
            {
                const inout = tracker.next(false);
                self.mats[0].mul_right_vec_dX(err, ptrs[inout[1]], .overwrite);
                try self.mats[0].mul_right_vec_dM(err, xs[0..n_dim], &out_dM.mats[0], strat);
            }

            for (self.mats[1 .. n_mat - 1], out_dM.mats[1 .. n_mat - 1], 1..) |*mat, *out_mat, i| {
                const inout = tracker.next(false);
                mat.mul_right_vec_dX(ptrs[inout[0]], ptrs[inout[1]], .overwrite);
                mat.mul_right_vec_dM(ptrs[inout[0]], xs[i * n_dim ..][0..n_dim], out_mat, strat);
            }

            {
                const mat = &self.mats[n_mat - 1];
                var out_mat = &out_dM.mats[n_mat - 1];
                const inout = tracker.next(true);
                mat.mul_right_vec_dX(ptrs[inout[0]], ptrs[inout[1]], strat);
                try mat.mul_right_vec_dM(ptrs[inout[0]], xs[(n_mat - 1) * n_dim ..][0..n_dim], out_mat, strat);
            }
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
    var buf: [8]f32 = undefined;
    mat.mul_left_vec(&x, &y, &buf);
    try revDeepEqual(y, .{ 132, 440, 63, 255 });
}

test "LogSquareMat mul_left_vec_dX" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const err: [4]f32 = .{ 1, 2, 3, 4 };
    var dX: [4]f32 = undefined;
    var buf: [8]f32 = undefined;
    mat.mul_left_vec_dX(&err, &dX, &buf, .overwrite);
    try revDeepEqual(dX, .{ 159, 247, 176, 260 });
    mat.mul_left_vec_dX(&err, &dX, &buf, .accumulate);
    try revDeepEqual(dX, .{ 159 * 2, 247 * 2, 176 * 2, 260 * 2 });
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
    var buf: [8]f32 = undefined;

    // check overwritten gradients are correct
    try mat.mul_left_vec_dM(&err, &xs, &grad, &buf, .overwrite);

    try revDeepEqual(xs, .{ 1, 2, 3, 4, 28, 33, 40, 9 });
    try revDeepEqual(y, .{ 132, 440, 63, 255 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);

    // check accumulated (now 2x) gradients are correct
    try mat.mul_left_vec_dM(&err, &xs, &grad, &buf, .accumulate);
    grad.scale_by(0.5);

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
    var buf: [8]f32 = undefined;

    // check overwritten gradients are correct
    try mat.mul_left_vec_dMdX(&err, &xs, &grad, &dX, &buf, .overwrite);

    try revDeepEqual(xs, .{ 1, 2, 3, 4, 28, 33, 40, 9 });
    try revDeepEqual(y, .{ 132, 440, 63, 255 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);

    try revDeepEqual(dX, .{ 159, 247, 176, 260 });

    // check accumulated (now 2x) gradients are correct
    try mat.mul_left_vec_dMdX(&err, &xs, &grad, &dX, &buf, .accumulate);
    grad.scale_by(0.5);

    try revDeepEqual(xs, .{ 1, 2, 3, 4, 28, 33, 40, 9 });
    try revDeepEqual(y, .{ 132, 440, 63, 255 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);

    try revDeepEqual(dX, .{ 159 * 2, 247 * 2, 176 * 2, 260 * 2 });
}

test "LogSquareMat mul_right_vec" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const x: [4]f32 = .{ 1, 2, 3, 4 };
    var y: [4]f32 = undefined;
    var buf: [8]f32 = undefined;
    mat.mul_right_vec(&x, &y, &buf);
    try revDeepEqual(y, .{ 159, 247, 176, 260 });
}

test "LogSquareMat mul_right_vec_dX" {
    const L = LogSquareMat(f32, 4, 2, 2);
    var mat: L = get_test_mat();
    const err: [4]f32 = .{ 1, 2, 3, 4 };
    var dX: [4]f32 = undefined;
    var buf: [8]f32 = undefined;
    mat.mul_right_vec_dX(&err, &dX, &buf, .overwrite);
    try revDeepEqual(dX, .{ 132, 440, 63, 255 });
    mat.mul_right_vec_dX(&err, &dX, &buf, .accumulate);
    try revDeepEqual(dX, .{ 132 * 2, 440 * 2, 63 * 2, 255 * 2 });
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
    var buf: [8]f32 = undefined;

    // check overwritten gradients are correct
    try mat.mul_right_vec_dM(&err, &xs, &grad, &buf, .overwrite);

    try revDeepEqual(xs, .{ 12, 16, 22, 53, 1, 2, 3, 4 });
    try revDeepEqual(y, .{ 159, 247, 176, 260 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);

    // check accumulated (now 2x) gradients are correct
    try mat.mul_right_vec_dM(&err, &xs, &grad, &buf, .accumulate);
    grad.scale_by(0.5);

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
    var buf: [8]f32 = undefined;

    // check overwritten gradients are correct
    try mat.mul_right_vec_dMdX(&err, &xs, &grad, &dX, &buf, .overwrite);

    try revDeepEqual(xs, .{ 12, 16, 22, 53, 1, 2, 3, 4 });
    try revDeepEqual(y, .{ 159, 247, 176, 260 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);

    try revDeepEqual(y, .{ 159, 247, 176, 260 });
    try revDeepEqual(dX, .{ 132, 440, 63, 255 });

    // check accumulated (now 2x) gradients are correct
    try mat.mul_right_vec_dMdX(&err, &xs, &grad, &dX, &buf, .accumulate);
    grad.scale_by(0.5);

    try revDeepEqual(xs, .{ 12, 16, 22, 53, 1, 2, 3, 4 });
    try revDeepEqual(y, .{ 159, 247, 176, 260 });

    try revDeepEqual(grad.mats[0].data, .{ 53, 53, 106, 44, 48, 48, 48, 88 });
    try revDeepEqual(grad.mats[1].data, .{ 112, 112, 132, 33, 80, 80, 27, 36 });

    try revEqual(grad.mats[0].mat_idx, mat.mats[0].mat_idx);
    try revEqual(grad.mats[1].mat_idx, mat.mats[1].mat_idx);

    try revDeepEqual(y, .{ 159, 247, 176, 260 });
    try revDeepEqual(dX, .{ 132 * 2, 440 * 2, 63 * 2, 255 * 2 });
}

test "LogSquareMat well-defined memory layout" {
    const L = LogSquareMat(f32, 3, 1, 2);
    const z: L = undefined;
    const bytes: [@sizeOf(L)]u8 = @bitCast(z);
    _ = bytes;
}
