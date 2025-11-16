const std = @import("std");
const zigimg = @import("zigimg");

///COLUMN MAJOR
fn SqMatrix(comptime d: usize) type {
    return struct {
        components: []f32,
        allocator: std.mem.Allocator,

        fn init(allocator: std.mem.Allocator) !SqMatrix(d) {
            const components = try allocator.alloc(f32, d * d);
            @memset(components, 0);
            return SqMatrix(d){
                .components = components,
                .allocator = allocator,
            };
        }

        fn deinit(self: *const @This()) void {
            self.allocator.free(self.components);
        }

        fn get(self: *const @This(), col: usize, row: usize) f32 {
            return self.components[col * d + row];
        }

        fn set(self: *const @This(), col: usize, row: usize, value: f32) void {
            self.components[col * d + row] = value;
        }

        fn mul(self: *const @This(), vec: *const Vector(d), allocator: std.mem.Allocator) !*Vector(d) {
            const out = try Vector(d).init(allocator);
            for (0..d) |i| {
                var sum: f32 = 0.0;
                for (0..d) |j| {
                    sum += vec.get(j) * self.get(i, j);
                }

                out.set(i, sum);
            }

            return out;
        }

        fn mulInto(self: *const @This(), vec: *const Vector(d), out: *Vector(d)) *Vector(d) {
            for (0..d) |i| {
                var sum: f32 = 0.0;
                for (0..d) |j| {
                    sum += vec.get(j) * self.get(i, j);
                }

                out.set(i, sum);
            }

            return out;
        }

        pub fn format(value: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;

            for (0..d) |y| {
                try writer.writeAll("[ ");
                for (0..d) |x| {
                    try writer.print("{e: >8.2}", .{value.get(y, x)});
                    if (x != d - 1) {
                        try writer.writeAll(", ");
                    }
                }
                try writer.writeAll(" ]\n");
            }
        }
    };
}

fn Vector(comptime d: usize) type {
    return struct {
        elements: []f32,
        allocator: std.mem.Allocator,

        fn init(allocator: std.mem.Allocator) !Vector(d) {
            const elements = try allocator.alloc(f32, d);
            @memset(elements, 0);
            return Vector(d){
                .elements = elements,
                .allocator = allocator,
            };
        }

        fn fromArray(arr: [d]f32, allocator: std.mem.Allocator) !@This() {
            const elements = try allocator.dupe(f32, arr[0..]);
            return Vector(d){
                .elements = elements,
                .allocator = allocator,
            };
        }

        fn clone(other: *const @This(), allocator: std.mem.Allocator) !@This() {
            const elements = try allocator.dupe(f32, other.elements);
            return Vector(d){
                .elements = elements,
                .allocator = allocator,
            };
        }

        fn deinit(self: *const @This()) void {
            self.allocator.free(self.elements);
        }

        fn get(self: *const @This(), idx: usize) f32 {
            return self.elements[idx];
        }

        fn set(self: *@This(), idx: usize, value: f32) void {
            self.elements[idx] = value;
        }

        fn add(self: *@This(), other: *@This()) *@This() {
            for (0..d) |i| {
                self.set(i, self.get(i) + other.get(i));
            }

            return self;
        }

        fn addArr(self: *@This(), slice: [d]f32) *@This() {
            for (0..d) |i| {
                self.set(i, self.get(i) + slice[i]);
            }

            return self;
        }

        fn subArr(self: *@This(), slice: [d]f32) *@This() {
            for (0..d) |i| {
                self.set(i, self.get(i) - slice[i]);
            }

            return self;
        }

        fn sub(self: *@This(), other: *@This()) *@This() {
            for (0..d) |i| {
                self.set(i, self.get(i) - other.get(i));
            }

            return self;
        }

        fn lengthSquared(self: *const @This()) f32 {
            var sum: f32 = 0.0;
            for (0..d) |i| {
                sum += self.get(i) * self.get(i);
            }

            return sum;
        }

        fn dot(self: *const @This(), other: *const @This()) f32 {
            var sum: f32 = 0.0;
            for (0..d) |i| {
                sum += self.get(i) * other.get(i);
            }

            return sum;
        }

        fn mulScalar(self: *@This(), scalar: f32) *@This() {
            for (0..d) |i| {
                self.set(i, self.get(i) * scalar);
            }

            return self;
        }

        pub fn format(value: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;

            try writer.writeAll("[ ");
            for (0..d) |i| {
                try writer.print("{e: >5.4}", .{value.get(i)});
                if (i != d - 1) {
                    try writer.writeAll(", ");
                }
            }

            try writer.writeAll(" ]");
        }
    };
}

/// solves Ax = b for b and returns x; A must be symmetric positive definite
/// start out with a b as the residual
/// p is the direction of travel
/// repeatedly:
///   move in the direction of the residual
///   adjust the residual based on the movement
///   if residual is small, we're done
///   update the search direction to be perpendicular to the previous (all) search directions
fn solveSystem(comptime size: usize, A: *const SqMatrix(size), b: *const Vector(size), allocator: std.mem.Allocator) !Vector(size) {
    const epsilon: f32 = 1e-11;
    var residual: Vector(size) = try Vector(size).clone(b, allocator);
    defer residual.deinit();

    var pos: Vector(size) = try Vector(size).init(allocator);
    if (residual.lengthSquared() < epsilon) return pos;

    var direction: Vector(size) = try Vector(size).clone(&residual, allocator);
    defer direction.deinit();

    var A_direction = try Vector(size).init(allocator);
    defer A_direction.deinit();

    var k: usize = 0;

    for (0..size) |_| {
        _ = A.mulInto(&direction, &A_direction);
        const a: f32 = residual.lengthSquared() / direction.dot(&A_direction);
        var step: [size]f32 = undefined;
        for (0..size) |i| {
            step[i] = direction.get(i) * a;
        }
        _ = pos.addArr(step);
        const old_residual_length_sq = residual.lengthSquared();
        var A_step: [size]f32 = undefined;
        for (0..size) |i| {
            A_step[i] = A_direction.get(i) * a;
        }
        _ = residual.subArr(A_step);
        const new_residual_length_sq = residual.lengthSquared();
        if (new_residual_length_sq < epsilon) return pos;
        const beta = new_residual_length_sq / old_residual_length_sq;
        _ = direction.mulScalar(beta).add(&residual);
        k += 1;
    }

    return pos;
}

/// mesh is broken up into triangles (elements); each triangle stores pointers to its nodes
/// stored as an AABB tree
const AABB = struct {
    min: Vector(2),
    max: Vector(2),

    fn unifiedCost(self: *const AABB, other: *const AABB) f32 {
        const min_x = @min(self.min.get(0), other.min.get(0));
        const min_y = @min(self.min.get(1), other.min.get(1));
        const max_x = @max(self.max.get(0), other.max.get(0));
        const max_y = @max(self.max.get(1), other.max.get(1));

        return (max_x - min_x) * (max_y - min_y);
    }
};

/// holds its aabb and indices for its nodes
const TriangleElement = struct {
    aabb: AABB,
    a: usize,
    b: usize,
    c: usize,

    fn init(a: usize, b: usize, c: usize, nodes: []Vector(2), allocator: std.mem.Allocator) !TriangleElement {
        const ap = nodes[a];
        const bp = nodes[b];
        const cp = nodes[c];

        const min_x: f32 = @min(ap.get(0), bp.get(0), cp.get(0));
        const min_y: f32 = @min(ap.get(1), bp.get(1), cp.get(1));
        const max_x: f32 = @max(ap.get(0), bp.get(0), cp.get(0));
        const max_y: f32 = @max(ap.get(1), bp.get(1), cp.get(1));

        return TriangleElement{
            .aabb = AABB{
                .min = try Vector(2).fromArray([2]f32{ min_x, min_y }, allocator),
                .max = try Vector(2).fromArray([2]f32{ max_x, max_y }, allocator),
            },
            .a = a,
            .b = b,
            .c = c,
        };
    }

    fn deinit(self: TriangleElement) void {
        self.aabb.min.deinit();
        self.aabb.max.deinit();
    }
};

pub fn test_main() !void {
    const allocator = std.heap.page_allocator;

    const m = try SqMatrix(3).init(allocator);
    defer m.deinit();

    m.set(1, 2, 69.0 / 17.0);

    var v = try Vector(3).init(allocator);
    defer v.deinit();

    v.set(2, 23.4);

    std.debug.print("m: \n{any}\n", .{m});
    std.debug.print("v: \n{any}\n", .{v});

    try test_mult();
    try test_mesh();
}

fn test_mult() !void {
    const allocator = std.heap.page_allocator;

    const m = try SqMatrix(3).init(allocator);
    defer m.deinit();

    m.set(0, 0, 4);
    m.set(0, 1, -4);
    m.set(0, 2, 0);

    m.set(1, 0, -4);
    m.set(1, 1, 4);
    m.set(1, 2, -4);

    m.set(2, 0, 0);
    m.set(2, 1, -4);
    m.set(2, 2, 4);

    var v = try Vector(3).init(allocator);
    defer v.deinit();

    v.set(0, -4);
    v.set(1, -8);
    v.set(2, 4);

    const x = try solveSystem(3, &m, &v, allocator);
    defer x.deinit();
    std.debug.print("x: \n{any}\n", .{x});
}

fn drawPixel(data: []u8, width: usize, height: usize, x: isize, y: isize, r: u8, g: u8, b: u8) void {
    _ = height;
    const x_u: usize = @intCast(x);
    const y_u: usize = @intCast(y);
    const idx = 3 * (width * y_u + x_u);
    if (idx < 0 or idx + 2 > data.len) return;
    data[idx] = r;
    data[idx + 1] = g;
    data[idx + 2] = b;
}

/// https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
fn drawLine(data: []u8, width: usize, height: usize, x0_: isize, y0_: isize, x1_: isize, y1_: isize, r: u8, g: u8, b: u8) void {
    var x0 = x0_;
    var y0 = y0_;
    const x1 = x1_;
    const y1 = y1_;
    const width_i: isize = @intCast(width);
    const height_i: isize = @intCast(height);
    const dx: isize = @intCast(@abs(x1 - x0));
    const sx: isize = if (x0 < x1) 1 else -1;
    var dy: isize = @intCast(@abs(y1 - y0));
    dy = -dy;
    const sy: isize = if (y0 < y1) 1 else -1;
    var err: isize = dx + dy;

    while (true) {
        if (x0 >= 0 and x0 < width_i and y0 >= 0 and y0 < height_i) {
            const idx: usize = @intCast(y0 * width_i + x0);
            data[3 * idx] = r;
            data[3 * idx + 1] = g;
            data[3 * idx + 2] = b;
        }

        const e2 = 2 * err;
        if (e2 >= dy) {
            if (x0 == x1) break;
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            if (y0 == y1) break;
            err += dx;
            y0 += sy;
        }
    }
}

const width_nodes = 64;
const height_nodes = 65;
const node_separation = 3;
const num_nodes = width_nodes * height_nodes;

fn constructNodes(allocator: std.mem.Allocator) ![]Vector(2) {
    var nodes: []Vector(2) = try allocator.alloc(Vector(2), num_nodes);
    for (0..height_nodes) |y| {
        const offset: usize = if (y % 2 == 0) 0 else node_separation / 2;
        for (0..width_nodes) |x| {
            nodes[y * width_nodes + x] = try Vector(2).fromArray([_]f32{ @floatFromInt(x * node_separation + offset), @floatFromInt(y * node_separation) }, allocator);
        }
    }

    return nodes;
}

/// k_{ij} = integral of basis'i * basis'j on element E
/// basis'i is a simplex s.t. basis'i(node_j) == kronocker_{ij}
/// a point p in a triangle can be described as c_a a + c_b b + c_c c
/// phi_i is basis function on node i, on element j, s.t. phi_i_j(C) = C[k]
/// c is linear so gradient of c is obviously constant, thus k_{ij} = gradient(phi_i(x)) * gradient(phi_j(x)) * area(E)
/// area(E) = det(V_1 - V_2, V_3 - V_2)/2
/// c1 = det(x - V_2, V_3 - V_2)/2area
/// this makes sense since it's linear in x, which is what we're looking for from the coefficient
/// derivative of c1 with respect to x works out to be (y_3 - y_2, x_2 - x_3)/2area
/// let b_i = (y_m - y_n, x_n - x_m), same for b_j
/// integral works out to b_i . b_j / 4area
///
/// algorithm:
///   out = matrix(size, size) = 0;
///   for element in elements:
///     A = area(element)
///     for node_i in element.nodes:
///         for node_j in element.nodes:
///             node_k = last node s.t. node_k != node_i != node_j
///             b_i = (node_j.y - node_k.y, node_k.x - node_j.x)
///             b_j = (node_i.y - node_k.y, node_k.x - node_i.x)
///             matrix[i, j] = b_i . b_j / (4A)
///     end for
///   end for
fn calculateStiffness(comptime size: usize, elements: []TriangleElement, nodes: []Vector(2), out: *SqMatrix(size)) void {
    for (elements) |element| {
        const element_node_indices = [_]usize{ element.a, element.b, element.c };
        const element_nodes = [_]Vector(2){ nodes[element.a], nodes[element.b], nodes[element.c] };
        const area = @abs(((element_nodes[0].get(0) - element_nodes[1].get(0)) * (element_nodes[2].get(1) - element_nodes[1].get(1)) - (element_nodes[0].get(1) - element_nodes[1].get(1)) * (element_nodes[2].get(0) - element_nodes[1].get(0))) / 2.0);
        for (0..3) |i| {
            const node_i_idx = element_node_indices[i];
            const node_i_n = element_nodes[(i + 1) % 3];
            const node_i_m = element_nodes[(i + 2) % 3];
            for (0..3) |j| {
                const node_j_idx = element_node_indices[j];
                const node_j_n = element_nodes[(j + 1) % 3];
                const node_j_m = element_nodes[(j + 2) % 3];

                const b_i_x = node_i_m.get(1) - node_i_n.get(1);
                const b_i_y = node_i_n.get(0) - node_i_m.get(0);
                const b_j_x = node_j_m.get(1) - node_j_n.get(1);
                const b_j_y = node_j_n.get(0) - node_j_m.get(0);

                // std.debug.print("i: {d} ({d}), j: {d} ({d})\n", .{ node_i_idx, i, node_j_idx, j });
                // std.debug.print("  node_i_n: {any}, node_i_m: {any}\n", .{ node_i_n, node_i_m });
                // std.debug.print("  node_j_n: {any}, node_j_m: {any}\n", .{ node_j_n, node_j_m });
                // std.debug.print("  db_i: {d}, {d}\n", .{ b_i_x / (2 * area), b_i_y / (2 * area) });
                // std.debug.print("  db_j: {d}, {d}\n", .{ b_j_x / (2 * area), b_j_y / (2 * area) });

                out.set(node_i_idx, node_j_idx, out.get(node_i_idx, node_j_idx) + (b_i_x * b_j_x + b_i_y * b_j_y) / (4 * area));
            }
        }
    }
}

fn applyBoundaryConditions(comptime size: usize, stiffness: *SqMatrix(size), load: *Vector(size)) void {
    //the corners of these setters overlap but it doesn't really matter
    for (0..width_nodes) |x| {
        // top row
        applyBoundryAtNode(size, stiffness, load, x, 0);
        // bottom row
        applyBoundryAtNode(size, stiffness, load, (height_nodes - 1) * width_nodes + x, 0);
    }

    for (0..height_nodes) |y| {
        const yf: f32 = @floatFromInt(y);
        // left column
        applyBoundryAtNode(size, stiffness, load, y * width_nodes, 0);
        // right column
        applyBoundryAtNode(size, stiffness, load, y * width_nodes + width_nodes - 1, std.math.sin(yf / 3) * 5000);
    }
}

fn applyBoundryAtNode(comptime size: usize, stiffness: *SqMatrix(size), load: *Vector(size), idx: usize, value: f32) void {
    load.set(idx, value);
    for (0..size) |y| {
        stiffness.set(idx, y, 0);
    }

    for (0..size) |x| {
        stiffness.set(x, idx, 0);
    }

    stiffness.set(idx, idx, 1);
}

fn evalForcing(x: f32, y: f32) f32 {
    return std.math.sin(std.math.pi * (x - std.math.log2(y) * 10) / 20.0) * 100;
}

/// gaussian quadrature
/// integral f * basis_i
/// iterate through every element
/// for an element e with vertices V1 V2 V3, mapping to (0,0), (1,0), (0,1)
/// it's obvious that: p(xi, eta) = V1 + (V2 - V1)xi + (V3 - V1)eta
/// xi >= 0, eta >= 0, xi + eta <= 1
/// so our integral becomes: integral of f(p(xi, eta)) * basis(xi, eta) * 2 * area(e)
/// we multiply by 2 * area(e) bcs = det(J) where J is the jacobian for this transformation; intuitively, the reference triangle has area 1/2 so multiplying it by 2 * area(e) denormalizes us back to the original domain
/// it's obvious that:
/// basis_0 = 1 - xi - eta
/// basis_1 = xi
/// basis_2 = eta
/// now we will sample at 3 points (xi, eta): (1/6, 1/6), (2/3, 1/6), (1/6, 2/3)
/// with a weight of 1/6 since our area of sampling must add up to 1/2 (the area of the reference triangle)
/// so essentially: 2 * area * sum (weight * f(p(xi, eta)) * basis(xi, eta))
fn calculateLoad(comptime size: usize, elements: []TriangleElement, nodes: []Vector(2), out: *Vector(size)) void {
    const weight: f32 = 1.0 / 6.0;
    const xi_a: f32 = 1.0 / 6.0;
    const eta_a: f32 = 1.0 / 6.0;
    const xi_b: f32 = 2.0 / 3.0;
    const eta_b: f32 = 1.0 / 6.0;
    const xi_c: f32 = 1.0 / 6.0;
    const eta_c: f32 = 2.0 / 3.0;

    // these are the basis functions i,j,k evaluated at points a,b,c
    const basis_i_a = 1 - xi_a - eta_a;
    const basis_i_b = 1 - xi_b - eta_b;
    const basis_i_c = 1 - xi_c - eta_c;
    const basis_j_a = xi_a;
    const basis_j_b = xi_b;
    const basis_j_c = xi_c;
    const basis_k_a = eta_a;
    const basis_k_b = eta_b;
    const basis_k_c = eta_c;

    for (elements) |element| {
        const node_i = nodes[element.a];
        const node_j = nodes[element.b];
        const node_k = nodes[element.c];

        const area = @abs(((node_i.get(0) - node_j.get(0)) * (node_k.get(1) - node_j.get(1)) - (node_i.get(1) - node_j.get(1)) * (node_k.get(0) - node_j.get(0))) / 2.0);

        const ax = node_i.get(0) + (node_j.get(0) - node_i.get(0)) * xi_a + (node_k.get(0) - node_i.get(0)) * eta_a;
        const ay = node_i.get(1) + (node_j.get(1) - node_i.get(1)) * xi_a + (node_k.get(1) - node_i.get(1)) * eta_a;
        const bx = node_i.get(0) + (node_j.get(0) - node_i.get(0)) * xi_b + (node_k.get(0) - node_i.get(0)) * eta_b;
        const by = node_i.get(1) + (node_j.get(1) - node_i.get(1)) * xi_b + (node_k.get(1) - node_i.get(1)) * eta_b;
        const cx = node_i.get(0) + (node_j.get(0) - node_i.get(0)) * xi_c + (node_k.get(0) - node_i.get(0)) * eta_c;
        const cy = node_i.get(1) + (node_j.get(1) - node_i.get(1)) * xi_c + (node_k.get(1) - node_i.get(1)) * eta_c;

        const f_a = evalForcing(ax, ay);
        const f_b = evalForcing(bx, by);
        const f_c = evalForcing(cx, cy);

        out.set(element.a, out.get(element.a) + 2 * area * weight * (f_a * basis_i_a + f_b * basis_i_b + f_c * basis_i_c));
        out.set(element.b, out.get(element.b) + 2 * area * weight * (f_a * basis_j_a + f_b * basis_j_b + f_c * basis_j_c));
        out.set(element.c, out.get(element.c) + 2 * area * weight * (f_a * basis_k_a + f_b * basis_k_b + f_c * basis_k_c));
    }
}

fn signedArea(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) f32 {
    return ((ax - bx) * (cy - by) - (ay - by) * (cx - bx)) / 2.0;
}

fn rasterize(elements: []TriangleElement, nodes: []Vector(2), coefficients: Vector(num_nodes), allocator: std.mem.Allocator) !void {
    const min_x: f32 = 0;
    const min_y: f32 = 0;
    const max_x: f32 = 256;
    const max_y: f32 = 256;

    const width = 256;
    const width_f: f32 = @floatFromInt(width);
    const height = 256;
    const height_f: f32 = @floatFromInt(height);

    const x_ratio = width_f / (max_x - min_x);
    const y_ratio = height_f / (max_y - min_y);

    const data = try allocator.alloc(u8, 3 * width * height);
    defer allocator.free(data);
    @memset(data, 255);

    var min: f32 = std.math.floatMax(f32);
    var max: f32 = -std.math.floatMax(f32);

    for (elements) |element| {
        const c_a = coefficients.get(element.a);
        const c_b = coefficients.get(element.b);
        const c_c = coefficients.get(element.c);

        const value = c_a * 0.5 + c_b * 0.5 + c_c * 0.5;

        min = @min(min, value);
        max = @max(max, value);
    }

    if (min > max) return;

    for (elements) |element| {
        const node_i = nodes[element.a];
        const node_j = nodes[element.b];
        const node_k = nodes[element.c];

        const c_a = coefficients.get(element.a);
        const c_b = coefficients.get(element.b);
        const c_c = coefficients.get(element.c);

        const area = @abs(signedArea(node_i.get(0), node_i.get(1), node_j.get(0), node_j.get(1), node_k.get(0), node_k.get(1)));

        const aabb_min_x = element.aabb.min.get(0);
        const aabb_min_y = element.aabb.min.get(1);
        const aabb_max_x = element.aabb.max.get(0);
        const aabb_max_y = element.aabb.max.get(1);

        const aabb_screen_min_x: isize = @intFromFloat(@round((aabb_min_x - min_x) * x_ratio));
        const aabb_screen_min_y: isize = @intFromFloat(@round((aabb_min_y - min_y) * y_ratio));
        const aabb_screen_max_x: isize = @intFromFloat(@round((aabb_max_x - min_x) * x_ratio));
        const aabb_screen_max_y: isize = @intFromFloat(@round((aabb_max_y - min_y) * y_ratio));

        var y = aabb_screen_min_y;
        while (y < aabb_screen_max_y) : (y += 1) {
            var yf: f32 = @floatFromInt(y);
            yf /= y_ratio;
            yf += min_y;
            var x = aabb_screen_min_x;
            while (x < aabb_screen_max_x) : (x += 1) {
                var xf: f32 = @floatFromInt(x);
                xf /= x_ratio;
                xf += min_x;

                const c_i = signedArea(xf, yf, node_k.get(0), node_k.get(1), node_j.get(0), node_j.get(1)) / area;
                const c_j = signedArea(xf, yf, node_i.get(0), node_i.get(1), node_k.get(0), node_k.get(1)) / area;
                const c_k = signedArea(xf, yf, node_j.get(0), node_j.get(1), node_i.get(0), node_i.get(1)) / area;

                if (c_i < 0 or c_j < 0 or c_k < 0) {
                    continue;
                }

                const value = c_a * c_i + c_b * c_j + c_c * c_k;
                const red: u8 = @intFromFloat(@max(0, @min(255, (value - min) / (max - min) * 255)));
                const blue: u8 = 255 - red;

                drawPixel(data, width, height, x, y, red, 0, blue);
            }
        }
    }

    var img = try zigimg.Image.fromRawPixels(allocator, width, height, data, .rgb24);
    defer img.deinit();

    try img.writeToFilePath("solution.qoi", .{ .qoi = .{} });
}

fn test_mesh() !void {
    var gpa = std.heap.DebugAllocator(.{}).init;
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) {
            @panic("Memory leak detected!");
        }
    }

    const allocator = gpa.allocator();

    {
        var timer = try std.time.Timer.start();
        const nodes = try constructNodes(allocator);
        defer {
            for (nodes) |node| {
                node.deinit();
            }

            allocator.free(nodes);
        }

        const num_triangles = (width_nodes - 1) * 2 * (height_nodes - 1);
        var triangles: [num_triangles]TriangleElement = undefined;

        var triangle_count: usize = 0;
        for (0..(height_nodes / 2)) |y| {
            for (0..(width_nodes - 1)) |x| {
                const a = x + width_nodes * y * 2;
                const b = x + width_nodes * y * 2 + 1;
                const c = x + width_nodes * (y * 2 + 1);
                const triangle = try TriangleElement.init(a, b, c, nodes, allocator);
                triangles[triangle_count] = triangle;
                triangle_count += 1;
            }

            for (0..(width_nodes - 1)) |x| {
                const a = x + width_nodes * (y * 2 + 1);
                const b = x + width_nodes * (y * 2 + 1) + 1;
                const c = x + width_nodes * (y * 2 + 2) + 1;
                const triangle = try TriangleElement.init(a, b, c, nodes, allocator);
                triangles[triangle_count] = triangle;
                triangle_count += 1;
            }

            for (1..width_nodes) |x| {
                const a = x + width_nodes * y * 2;
                const b = x + width_nodes * (y * 2 + 1);
                const c = x + width_nodes * (y * 2 + 1) - 1;
                const triangle = try TriangleElement.init(a, b, c, nodes, allocator);
                triangles[triangle_count] = triangle;
                triangle_count += 1;
            }

            for (0..(width_nodes - 1)) |x| {
                const a = x + width_nodes * (y * 2 + 1);
                const b = x + width_nodes * (y * 2 + 2) + 1;
                const c = x + width_nodes * (y * 2 + 2);
                const triangle = try TriangleElement.init(a, b, c, nodes, allocator);
                triangles[triangle_count] = triangle;
                triangle_count += 1;
            }
        }

        std.debug.assert(num_triangles == triangle_count);

        defer {
            for (triangles) |triangle| {
                triangle.deinit();
            }
        }

        const elements_generation_duration = timer.lap();

        var stiffness = try SqMatrix(num_nodes).init(allocator);
        defer stiffness.deinit();

        calculateStiffness(num_nodes, &triangles, nodes, &stiffness);

        const stiffness_generation_duration = timer.lap();

        var load = try Vector(num_nodes).init(allocator);
        defer load.deinit();

        calculateLoad(num_nodes, &triangles, nodes, &load);

        const load_generation_duration = timer.lap();

        applyBoundaryConditions(num_nodes, &stiffness, &load);

        const boundary_condition_duration = timer.lap();

        const coefficients = try solveSystem(num_nodes, &stiffness, &load, allocator);
        defer coefficients.deinit();

        const solve_duration = timer.lap();

        // std.debug.print("stiffness: \n{any}\n", .{stiffness});
        // std.debug.print("load: {any}\n", .{load});
        // std.debug.print("coefficients: {any}\n", .{coefficients});

        try rasterize(&triangles, nodes, coefficients, allocator);

        const rasterization_duration = timer.lap();

        std.debug.print("element_generation_duration: {d}\n", .{elements_generation_duration / 1000});
        std.debug.print("stiffness_generation_duration: {d}\n", .{stiffness_generation_duration / 1000});
        std.debug.print("load_generation_duration: {d}\n", .{load_generation_duration / 1000});
        std.debug.print("boundary_condition_duration: {d}\n", .{boundary_condition_duration / 1000});
        std.debug.print("solve_duration: {d}\n", .{solve_duration / 1000});
        std.debug.print("rasterization_duration: {d}\n", .{rasterization_duration / 1000});
    }
}
