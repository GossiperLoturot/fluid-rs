use glam::*;
use rand::Rng;

const DT: f32 = 0.2;
const ITERATIONS: i32 = (1.0 / DT) as i32;

const PARTICLE_SIZE: i32 = 1024;
const GRID_RES: i32 = 64;

const GRAVITY: Vec2 = Vec2::new(0.0, -0.3);
const REST_DENSITY: f32 = 4.0;
const DYNAMIC_VISCOSITY: f32 = 0.1;
const EOS_STIFFNESS: f32 = 10.0;
const EOS_POWER: f32 = 4.0;

const STDOUT_SIZE: IVec2 = IVec2::new(80, 40);

#[derive(Default, Clone, Copy)]
struct Particle {
    x: Vec2,
    v: Vec2,
    c_mat: Mat2,
    m: f32,
}

#[derive(Default, Clone, Copy)]
struct Cell {
    v: Vec2,
    m: f32,
}

// pipeline

fn clear_grid(grid: &mut [Cell]) {
    for i in 0..GRID_RES * GRID_RES {
        grid[i as usize].v = Vec2::ZERO;
        grid[i as usize].m = 0.0;
    }
}

fn p2g_1(particles: &[Particle], grid: &mut [Cell]) {
    for i in 0..PARTICLE_SIZE {
        let p = &particles[i as usize];

        let cell_idx = p.x.floor().as_ivec2();
        let cell_diff = p.x - cell_idx.as_vec2() - Vec2::splat(0.5);

        let w = [
            0.5 * (0.5 - cell_diff).powf(2.0),
            0.75 - cell_diff.powf(2.0),
            0.5 * (0.5 + cell_diff).powf(2.0),
        ];

        for gy in 0..3 {
            for gx in 0..3 {
                let weight = w[gx].x * w[gy].y;

                let cell_x = cell_idx + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_x.as_vec2() - p.x + Vec2::splat(0.5);
                let q = p.c_mat * cell_dist;

                let mass_contrib = weight * p.m;

                let cell = &mut grid[(cell_x.y * GRID_RES + cell_x.x) as usize];
                cell.m += mass_contrib;
                cell.v += mass_contrib * (p.v + q);
            }
        }
    }
}

fn p2g_2(particles: &[Particle], grid: &mut [Cell]) {
    for i in 0..PARTICLE_SIZE {
        let p = &particles[i as usize];

        let cell_idx = p.x.floor().as_ivec2();
        let cell_diff = p.x - cell_idx.as_vec2() - Vec2::splat(0.5);

        let w = [
            0.5 * (0.5 - cell_diff).powf(2.0),
            0.75 - cell_diff.powf(2.0),
            0.5 * (0.5 + cell_diff).powf(2.0),
        ];

        let mut density = 0.0;
        for gy in 0..3 {
            for gx in 0..3 {
                let weight = w[gx].x * w[gy].y;

                let cell_x = cell_idx + IVec2::new(gx as i32 - 1, gy as i32 - 1);

                density += grid[(cell_x.y * GRID_RES + cell_x.x) as usize].m * weight;
            }
        }
        let volume = p.m / density;

        let pressure = f32::max(
            -0.1,
            EOS_STIFFNESS * ((density / REST_DENSITY).powf(EOS_POWER) - 1.0),
        );

        let mut strain = p.c_mat;
        let trace = strain.y_axis.x + strain.x_axis.y;
        strain.y_axis.x = trace;
        strain.x_axis.y = trace;

        let viscosity_term = DYNAMIC_VISCOSITY * strain;
        let stress = -pressure * Mat2::IDENTITY + viscosity_term;

        let eg_16_term_0 = 4.0 * -volume * stress * DT;

        for gy in 0..3 {
            for gx in 0..3 {
                let weight = w[gx].x * w[gy].y;

                let cell_x = cell_idx + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_x.as_vec2() - p.x + Vec2::splat(0.5);

                let cell = &mut grid[(cell_x.y * GRID_RES + cell_x.x) as usize];
                cell.v += weight * eg_16_term_0 * cell_dist;
            }
        }
    }
}

fn update_grid(grid: &mut [Cell]) {
    for i in 0..GRID_RES * GRID_RES {
        let cell = &mut grid[i as usize];

        if cell.m > 0.0 {
            cell.v /= cell.m;
            cell.v += DT * GRAVITY;

            let x = i % GRID_RES;
            if x < 2 || x > (GRID_RES - 1) - 2 {
                cell.v.x = 0.0;
            }

            let y = i / GRID_RES;
            if y < 2 || y > (GRID_RES - 1) - 2 {
                cell.v.y = 0.0;
            }
        }
    }
}

fn g2p(particles: &mut [Particle], grid: &[Cell]) {
    for i in 0..PARTICLE_SIZE {
        let p = &mut particles[i as usize];

        p.v = Vec2::ZERO;

        let cell_idx = p.x.floor().as_ivec2();
        let cell_diff = p.x - cell_idx.as_vec2() - Vec2::splat(0.5);

        let w = [
            0.5 * (0.5 - cell_diff).powf(2.0),
            0.75 - cell_diff.powf(2.0),
            0.5 * (0.5 + cell_diff).powf(2.0),
        ];

        let mut b_mat = Mat2::ZERO;
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = w[gx].x * w[gy].y;

                let cell_x = cell_idx + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_x.as_vec2() - p.x + Vec2::splat(0.5);

                let weighted_velocity = weight * grid[(cell_x.y * GRID_RES + cell_x.x) as usize].v;
                let term = Mat2::from_cols(
                    weighted_velocity * cell_dist.x,
                    weighted_velocity * cell_dist.y,
                );
                b_mat += term;

                p.v += weighted_velocity;
            }
        }

        p.c_mat = 4.0 * b_mat;
        p.x += p.v * DT;

        // boundary conditions

        p.x = Vec2::clamp(
            p.x,
            Vec2::splat(1.0),
            IVec2::splat(GRID_RES - 1).as_vec2() - Vec2::splat(1.0),
        );

        let x_n = p.x + p.v;
        let wall_min = 3.0;
        let wall_max = (GRID_RES - 1) as f32 - 3.0;
        if x_n.x < wall_min {
            p.v.x += wall_min - x_n.x;
        }
        if x_n.x > wall_max {
            p.v.x += wall_max - x_n.x;
        }
        if x_n.y < wall_min {
            p.v.y += wall_min - x_n.y;
        }
        if x_n.y > wall_max {
            p.v.y += wall_max - x_n.y;
        }
    }
}

// renderer

fn draw(particles: &[Particle]) {
    let mut bin_counts = [0; (STDOUT_SIZE.x * STDOUT_SIZE.y) as usize];

    for i in 0..PARTICLE_SIZE {
        let xyz = (particles[i as usize].x / Vec2::splat(GRID_RES as f32) * STDOUT_SIZE.as_vec2())
            .as_ivec2();

        if xyz.cmpge(IVec2::ZERO).all() && xyz.cmplt(STDOUT_SIZE).all() {
            bin_counts[(xyz.y * STDOUT_SIZE.x + xyz.x) as usize] += 1;
        }
    }

    print!("\x1b[2J");
    print!("\x1b[1;1H");
    for y in 0..STDOUT_SIZE.y {
        for x in 0..STDOUT_SIZE.x {
            match bin_counts[(y * STDOUT_SIZE.x + x) as usize] {
                n if n < 1 => print!(" "),
                n if n < 2 => print!("."),
                n if n < 3 => print!("-"),
                n if n < 4 => print!("="),
                n if n < 5 => print!("*"),
                n if n < 6 => print!("%"),
                n if n < 7 => print!("$"),
                _ => print!("#"),
            };
        }
        println!();
    }
}

// main routine

fn main() {
    let mut rng = rand::thread_rng();

    let mut particles = [Particle::default(); PARTICLE_SIZE as usize];
    let mut grid = [Cell::default(); (GRID_RES * GRID_RES) as usize];

    for i in 0..PARTICLE_SIZE {
        let x = rng.gen_range(24.0..=40.0);
        let y = rng.gen_range(24.0..=40.0);
        particles[i as usize].x = Vec2::new(x, y);
        particles[i as usize].v = Vec2::ZERO;
        particles[i as usize].c_mat = Mat2::ZERO;
        particles[i as usize].m = 1.0;
    }

    for i in 0..GRID_RES * GRID_RES {
        grid[i as usize].v = Vec2::ZERO;
        grid[i as usize].m = 0.0;
    }

    let time = std::time::Duration::from_secs_f32(DT);
    loop {
        draw(&particles);

        for _ in 0..ITERATIONS {
            clear_grid(&mut grid);
            p2g_1(&particles, &mut grid);
            p2g_2(&particles, &mut grid);
            update_grid(&mut grid);
            g2p(&mut particles, &grid);
        }

        std::thread::sleep(time);
    }
}
