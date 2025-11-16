const std = @import("std");
const fem_2d = @import("fem_2d.zig");
const zigimg = @import("zigimg");

const num_elements: i32 = 32;
const trapezoid_sum_samples: usize = 20;

// const Matrix = [num_elements - 1][num_elements - 1]f32;
//
const Matrix = struct {
    components: [num_elements - 1][num_elements - 1]f32,

    fn zero() Matrix {
        return Matrix{ .components = [_][num_elements - 1]f32{
            [_]f32{0} ** (num_elements - 1),
        } ** (num_elements - 1) };
    }

    fn mul(self: Matrix, vec: []f32, allocator: std.mem.Allocator) ![]f32 {
        const out = try allocator.alloc(f32, vec.len);
        const components = self.components;
        for (0..vec.len) |i| {
            var s: f32 = 0;
            for (0..vec.len) |j| {
                s += components[i][j] * vec[j];
            }
            out[i] = s;
        }

        return out;
    }
};

fn populateNodes(out: *[num_elements + 1]f32) void {
    const as_float: f32 = @floatFromInt(num_elements);
    const h = 1 / as_float;
    for (0..(num_elements + 1)) |i| {
        const i_f: f32 = @floatFromInt(i);
        out[i] = i_f * h;
    }
}

fn forcing(x: f32) f32 {
    return 100 * @sin(8 * std.math.pi * x) * std.math.pow(f32, std.math.e, -20 * (x - 0.5) * (x - 0.5));
}

/// i is 1-based since it corresponds to the node at the peak of each tent
fn basis(i: usize, x: f32, nodes: []f32) f32 {
    var idx: usize = 0;
    while (idx < nodes.len - 1 and nodes[idx + 1] <= x) idx += 1;
    if (idx == i - 1) {
        return (x - nodes[i - 1]) / (nodes[i] - nodes[i - 1]);
    } else if (idx == i) {
        return (nodes[i + 1] - x) / (nodes[i + 1] - nodes[i]);
    } else {
        return 0;
    }
}

fn dbasis(i: usize, x: f32, nodes: []f32) f32 {
    var idx: usize = 0;
    while (idx < nodes.len - 1 and nodes[idx + 1] <= x) idx += 1;
    if (idx == i - 1) {
        return 1 / (nodes[i] - nodes[i - 1]);
    } else if (idx == i) {
        return -1 / (nodes[i + 1] - nodes[i]);
    } else {
        return 0;
    }
}

fn calculateStiffness(out_k: *Matrix) void {
    const nef: f32 = @floatFromInt(num_elements);
    for (0..(num_elements - 1)) |e| {
        out_k.components[e][e] = 2 * nef;
        if (e < num_elements - 2) {
            out_k.components[e + 1][e] = -nef;
            out_k.components[e][e + 1] = -nef;
        }
    }
}

fn calculateLoad(nodes: []f32, out_b: *[num_elements - 1]f32) void {
    const nef: f32 = @floatFromInt(num_elements);
    const samples_f: f32 = @floatFromInt(trapezoid_sum_samples);
    const h = 1 / nef;
    const h_s = h / (samples_f + 1) * 2;
    for (0..(num_elements - 1)) |e| {
        var sum = forcing(nodes[e]) * basis(e + 1, nodes[e], nodes[0..]) + forcing(nodes[e + 2]) * basis(e + 1, nodes[e + 2], nodes[0..]);
        for (0..trapezoid_sum_samples) |s| {
            const sf: f32 = @floatFromInt(s);
            const x = nodes[e] + (sf + 1) * h_s;
            const y = forcing(x) * basis(e + 1, x, nodes[0..]);
            sum += 2 * y;
        }

        out_b[e] = 0.5 * h_s * sum;
    }
}

//ai'd because idgaf about writing this rn
fn solveLinearSystem(allocator: std.mem.Allocator, a: *[num_elements - 1][num_elements - 1]f32, b: []const f32) ![]f32 {
    const m = a.len;
    if (m == 0) return try allocator.alloc(f32, 0);
    if (b.len != m) return error.InvalidDimensions;
    const vars = if (m > 0) a[0].len else 0;
    for (a) |row| {
        if (row.len != vars) return error.InvalidDimensions;
    }
    var augmented = try allocator.alloc([]f32, m);
    defer {
        for (augmented) |row| {
            allocator.free(row);
        }
        allocator.free(augmented);
    }
    for (0..m) |i| {
        augmented[i] = try allocator.alloc(f32, vars + 1);
        @memcpy(augmented[i][0..vars], a[i][0..]);
        augmented[i][vars] = b[i];
    }
    var h: usize = 0;
    var k: usize = 0;
    while (h < m and k < vars) {
        var i_max: usize = h;
        var max_abs: f32 = @abs(augmented[h][k]);
        var i: usize = h + 1;
        while (i < m) : (i += 1) {
            const abs_val: f32 = @abs(augmented[i][k]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
                i_max = i;
            }
        }
        if (max_abs == 0.0) {
            k += 1;
            continue;
        }
        if (i_max != h) {
            const temp = augmented[h];
            augmented[h] = augmented[i_max];
            augmented[i_max] = temp;
        }
        i = h + 1;
        while (i < m) : (i += 1) {
            const f: f32 = augmented[i][k] / augmented[h][k];
            augmented[i][k] = 0.0;
            var j: usize = k + 1;
            while (j < vars + 1) : (j += 1) {
                augmented[i][j] -= augmented[h][j] * f;
            }
        }
        h += 1;
        k += 1;
    }
    {
        var i: usize = h;
        while (i < m) : (i += 1) {
            if (@abs(augmented[i][vars]) > 1e-10) {
                return error.NoSolution;
            }
        }
    }
    var solution = try allocator.alloc(f32, vars);
    for (solution) |*val| {
        val.* = 0.0;
    }
    var i: usize = if (h > 0) h - 1 else return solution;
    while (true) {
        var j: usize = 0;
        while (j < vars and @abs(augmented[i][j]) < 1e-10) : (j += 1) {}
        if (j == vars) unreachable; // should not happen
        var sum: f32 = augmented[i][vars];
        var col: usize = j + 1;
        while (col < vars) : (col += 1) {
            sum -= augmented[i][col] * solution[col];
        }
        solution[j] = sum / augmented[i][j];
        if (i == 0) break;
        i -= 1;
    }
    return solution;
}

fn imageTest(allocator: std.mem.Allocator) !void {
    var image = try zigimg.Image.fromFilePath(allocator, "image.png");
    defer image.deinit();

    const first_pixel = image.pixels.rgba32[0];
    std.debug.print("rgb: {d} {d} {d}\n", .{ first_pixel.r, first_pixel.g, first_pixel.b });

    std.debug.print("height: {d}\n", .{image.height});
}

fn createImageTest(allocator: std.mem.Allocator) !void {
    const width = 1024;
    const height = 1024;
    const data = try allocator.alloc(u8, 3 * width * height);
    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const r = prng.random();
    std.Random.bytes(r, data);
    var img = try zigimg.Image.fromRawPixels(allocator, width, height, data, .rgb24);
    defer img.deinit();

    try img.writeToFilePath("created_image.qoi", .{ .qoi = .{} });
}

fn createImageCosTest(allocator: std.mem.Allocator) !void {
    const width: usize = 512;
    const height: usize = 512;
    const data = try allocator.alloc(u8, 3 * width * height);
    @memset(data, 255);

    for (0..width) |x| {
        const xf: f32 = @floatFromInt(x);
        const raw_y = @sin(xf * 0.1);
        const rounded_yf: f32 = @round(raw_y * 10);
        const rounded_y: isize = @intFromFloat(rounded_yf);
        const screen_y: usize = @intCast(rounded_y + height / 2);
        const screen_idx = 3 * (screen_y * width + x);
        data[screen_idx] = 255;
        data[screen_idx + 1] = 0;
        data[screen_idx + 2] = 0;
    }

    var img = try zigimg.Image.fromRawPixels(allocator, width, height, data, .rgb24);
    defer img.deinit();

    try img.writeToFilePath("created_image.qoi", .{ .qoi = .{} });
}

const Graph = struct {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    function: *const fn (*const anyopaque, f32) f32,
    context: *const anyopaque,

    fn toImage(
        self: Graph,
        width: usize,
        height: usize,
        path: []const u8,
        allocator: std.mem.Allocator,
    ) !void {
        const wi: isize = @intCast(height);
        const wif: f32 = @floatFromInt(wi);
        const hi: isize = @intCast(height);
        const hif: f32 = @floatFromInt(hi);

        const delta_x = self.max_x - self.min_x;
        const dx_per_pixel = delta_x / wif;

        const delta_y = self.max_y - self.min_y;
        const pixels_per_dy = hif / delta_y;

        const data = try allocator.alloc(u8, 3 * width * height);
        @memset(data, 255);

        for (0..width) |x| {
            const xf: f32 = @floatFromInt(x);
            const raw_y = self.function(self.context, self.min_x + xf * dx_per_pixel);
            const rounded_yf: f32 = @round((raw_y - self.min_y) * pixels_per_dy);
            const rounded_y: isize = @intFromFloat(rounded_yf);
            if (rounded_y < 0 or rounded_y >= height) continue;
            var screen_y: usize = @intCast(rounded_y);
            screen_y = height - screen_y;
            if (screen_y * width + x < 0 or screen_y * width + x >= height * width) continue;
            const screen_idx = 3 * (screen_y * width + x);
            data[screen_idx] = 255;
            data[screen_idx + 1] = 0;
            data[screen_idx + 2] = 0;
        }

        var img = try zigimg.Image.fromRawPixels(allocator, width, height, data, .rgb24);
        defer img.deinit();

        try img.writeToFilePath(path, .{ .qoi = .{} });
    }
};

fn evalSolution(
    weights: []f32,
    nodes: []f32,
    x: f32,
) f32 {
    var sum: f32 = 0;
    for (1..num_elements) |i| {
        sum += weights[i - 1] * basis(i, x, nodes);
    }

    return sum;
}

pub fn main() !void {
    // const allocator = std.heap.page_allocator;
    // std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    // std.debug.print("num_elements: {d}, forcing(1) => {d}\n", .{ num_elements, forcing(1.0) });
    //
    // var nodes = [_]f32{0} ** (num_elements + 1);
    // populateNodes(&nodes);
    // const nodes_immutable = nodes[0..];
    //
    // var k = Matrix.zero();
    // calculateStiffness(&k);
    // std.debug.print("k: {any}\n", .{k});
    //
    // var b = [_]f32{0} ** (num_elements - 1);
    // calculateLoad(nodes[0..], &b);
    // std.debug.print("b: {any}\n", .{b});
    //
    // const solution = try solveLinearSystem(allocator, &k.components, &b);
    // // const solution = try solveLinearSystem(allocator, &[_][]const f32{
    // //     k.components[0][0..],
    // //     k.components[1][0..],
    // //     k.components[2][0..],
    // // }, &b);
    // std.debug.print("solution: {any}\n", .{solution});
    //
    // const computed_b = try k.mul(solution, allocator);
    // defer allocator.free(computed_b);
    //
    // for (0..b.len) |i| {
    //     std.debug.print("error: {d}\n", .{b[i] - computed_b[i]});
    // }
    //
    // try imageTest(allocator);
    // try createImageCosTest(allocator);
    //
    // // const forcing_graph = Graph{ .min_x = 0, .max_x = 1, .min_y = -2, .max_y = 2, .function = forcing };
    // //
    // // try forcing_graph.toImage(256, 256, "forcing_image.qoi", allocator);
    //
    // const Tent = struct {
    //     wts: []f32,
    //     nds: []f32,
    //
    //     pub fn eval(self: *const anyopaque, x: f32) f32 {
    //         const t: *const @This() = @ptrCast(@alignCast(self));
    //         return evalSolution(t.wts, t.nds, x);
    //     }
    // };
    //
    // const t: *const anyopaque = &Tent{ .wts = solution, .nds = nodes_immutable };
    //
    // const tent_graph = Graph{ .min_x = 0, .max_x = 1, .min_y = -2, .max_y = 2, .function = Tent.eval, .context = t };
    //
    // try tent_graph.toImage(256, 256, "tent_image.qoi", allocator);
    //
    try fem_2d.test_main();
}
