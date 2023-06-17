use glam::*;
use rand::Rng;

const PARTICLE_SIZE: usize = 1024;
const PARTICLE_RADIUS: f32 = 0.012;
const PARTICLE_MASS: f32 = 0.0002;
const VISCOSITY: f32 = 0.1;
const PRESSURE_STIFFNESS: f32 = 200.0;
const REST_DENSITY: f32 = 1000.0;
const TIME_STEP: f32 = 0.01666;
const WALL_STIFNESS: f32 = 3000.0;
const GRAVITY_ACCELERATION: f32 = 9.8;
const RANGE_X: f32 = 1.0;
const RANGE_Y: f32 = 1.0;
const PI: f32 = std::f32::consts::PI;

#[derive(Debug, Default, Clone, Copy)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
    density: f32,
    pressure: f32,
    force: Vec2,
}

fn poly6(r: Vec2, h: f32) -> f32 {
    if r.length() < h {
        4.0 / (PI * h.powi(8)) * (h.powi(2) - r.length_squared()).powi(3)
    } else {
        0.0
    }
}

fn lap_viscocity(r: Vec2, h: f32) -> f32 {
    if r.length() < h {
        20.0 / (3.0 * PI * h.powi(5)) * (h - r.length())
    } else {
        0.0
    }
}

fn grad_spiky(r: Vec2, h: f32) -> Vec2 {
    if r.length() < h {
        -30.0 / (PI * h.powi(5)) * (h - r.length()).powi(2) * r.normalize()
    } else {
        Vec2::ZERO
    }
}

fn compute_density(read_buffer: &[Particle], write_buffer: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        write_buffer[i].density = 0.0;

        for j in 0..PARTICLE_SIZE {
            if i == j {
                continue;
            }

            write_buffer[i].density += PARTICLE_MASS
                * poly6(
                    read_buffer[j].position - read_buffer[i].position,
                    PARTICLE_RADIUS,
                );
        }
    }
}

fn compute_pressure(read_buffer: &[Particle], write_buffer: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        write_buffer[i].pressure =
            PRESSURE_STIFFNESS * ((read_buffer[i].density / REST_DENSITY).powi(7) - 1.0).max(0.0);
    }
}

fn compute_force(read_buffer: &[Particle], write_buffer: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        write_buffer[i].force = Vec2::ZERO;

        for j in 0..PARTICLE_SIZE {
            if i == j {
                continue;
            }

            // viscosity force
            write_buffer[i].force +=
                VISCOSITY * PARTICLE_MASS * (read_buffer[j].velocity - read_buffer[i].velocity)
                    / read_buffer[j].density
                    * lap_viscocity(
                        read_buffer[j].position - read_buffer[i].position,
                        PARTICLE_RADIUS,
                    );

            // pressure force
            write_buffer[i].force += -1.0 / read_buffer[i].density
                * PARTICLE_MASS
                * (read_buffer[j].pressure - read_buffer[i].pressure)
                / (2.0 * read_buffer[j].density)
                * grad_spiky(
                    read_buffer[j].position - read_buffer[i].position,
                    PARTICLE_RADIUS,
                );
        }
    }
}

fn compute_position_and_velocity(read_buffer: &[Particle], write_buffer: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        let external_acceleration = Vec2::NEG_Y * GRAVITY_ACCELERATION;

        let mut penalty_acceleration = Vec2::ZERO;
        penalty_acceleration +=
            read_buffer[i].position.dot(Vec2::NEG_X).max(0.0) * WALL_STIFNESS * Vec2::X;
        penalty_acceleration +=
            (read_buffer[i].position.dot(Vec2::X) - RANGE_X).max(0.0) * WALL_STIFNESS * Vec2::NEG_X;
        penalty_acceleration +=
            read_buffer[i].position.dot(Vec2::NEG_Y).max(0.0) * WALL_STIFNESS * Vec2::Y;
        penalty_acceleration +=
            (read_buffer[i].position.dot(Vec2::Y) - RANGE_Y).max(0.0) * WALL_STIFNESS * Vec2::NEG_Y;

        let mut acceleration = read_buffer[i].force / read_buffer[i].density;
        if acceleration.is_nan() {
            acceleration = Vec2::ZERO;
        }

        let acceleration = acceleration + external_acceleration + penalty_acceleration;
        let velocity = read_buffer[i].velocity + acceleration * TIME_STEP;
        let position = read_buffer[i].position + velocity * TIME_STEP;

        write_buffer[i].velocity = velocity;
        write_buffer[i].position = position;
    }
}

fn render_to_cui(read_buffer: &[Particle]) {
    let mut text = String::new();
    let mut amount_map = [[0; 10]; 10];

    for i in 0..PARTICLE_SIZE {
        let position = read_buffer[i].position;

        let x = (position.x / RANGE_X * 10.0).floor() as i32;
        let y = (position.y / RANGE_Y * 10.0).floor() as i32;
        if 0 <= x && x < 10 && 0 <= y && y < 10 {
            amount_map[y as usize][x as usize] += 1;
        }
    }

    for i in 0..10 {
        for j in 0..10 {
            let amount = amount_map[i][j];
            text.push(if amount == 0 {
                ' '
            } else if amount < 16 {
                '.'
            } else if amount < 32 {
                '-'
            } else if amount < 64 {
                '='
            } else if amount < 128 {
                '*'
            } else if amount < 256 {
                '%'
            } else if amount < 512 {
                '$'
            } else {
                '#'
            });
        }
        text.push('\n');
    }

    print!("\x1b[2J\x1b[1;1H{}", text);
}

fn main() {
    let mut rng = rand::thread_rng();

    let mut init_buffer = [Particle::default(); PARTICLE_SIZE];
    for i in 0..PARTICLE_SIZE {
        init_buffer[i].position =
            Vec2::new(rng.gen_range(0.0..=RANGE_X), rng.gen_range(0.0..=RANGE_Y));
    }

    let mut read_buffer = init_buffer.clone();
    let mut write_buffer = init_buffer;

    loop {
        compute_density(&read_buffer, &mut write_buffer);
        read_buffer.copy_from_slice(&write_buffer);

        compute_pressure(&read_buffer, &mut write_buffer);
        read_buffer.copy_from_slice(&write_buffer);

        compute_force(&read_buffer, &mut write_buffer);
        read_buffer.copy_from_slice(&write_buffer);

        compute_position_and_velocity(&read_buffer, &mut write_buffer);
        read_buffer.copy_from_slice(&write_buffer);

        render_to_cui(&read_buffer);

        std::thread::sleep(std::time::Duration::from_secs_f32(TIME_STEP));
    }
}
