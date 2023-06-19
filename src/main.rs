use glam::*;
use rand::Rng;

// simulation params
const ITERATE: usize = 256;
const TIME_STEP: f32 = 0.0001;
const PARTICLE_SIZE: usize = 256;
const PARTICLE_RADIUS: f32 = 0.1;
const PARTICLE_MASS: f32 = 1.0;

// viscosity params
const VISCOSITY: f32 = 0.15;

// pressure params
const PRESSURE_STIFFNESS: f32 = 50.0;
const REST_DENSITY: f32 = 500.0;

// bounds params
const VELOCITY_DUMPING: f32 = -0.5;
const RANGE_X: f32 = 1.0;
const RANGE_Y: f32 = 1.0;

// renderer params
const STDOUT_X: usize = 32;
const STDOUT_Y: usize = 16;

// constant
const PI: f32 = std::f32::consts::PI;
const GRAVITY_ACCELERATION: f32 = 9.8;

#[derive(Debug, Default, Clone, Copy)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
    acceleration: Vec2,
    density: f32,
    pressure: f32,
}

// kernel functions

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

// pipeline functions
// refered from "Becker and Teschner, Weakly compressible SPH for free surface flows, SCA2007."

fn compute_density_and_pressure(particles: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        particles[i].density = 0.0;

        for j in 0..PARTICLE_SIZE {
            if i == j {
                continue;
            }

            // density
            particles[i].density += PARTICLE_MASS
                * poly6(
                    particles[i].position - particles[j].position,
                    PARTICLE_RADIUS,
                );
        }

        // pressure
        particles[i].pressure =
            PRESSURE_STIFFNESS * ((particles[i].density / REST_DENSITY).powi(7) - 1.0).max(0.0);
    }
}

fn compute_acceleration(particles: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        particles[i].acceleration = Vec2::ZERO;

        for j in 0..PARTICLE_SIZE {
            if i == j {
                continue;
            }

            // pressure
            let acceleration = -PARTICLE_MASS
                * (particles[i].pressure / particles[i].density.powi(2)
                    + particles[j].pressure / particles[j].density.powi(2))
                * grad_spiky(
                    particles[i].position - particles[j].position,
                    PARTICLE_RADIUS,
                );
            if !acceleration.is_nan() {
                particles[i].acceleration += acceleration;
            }

            // viscosity
            let acceleration =
                VISCOSITY * PARTICLE_MASS * (particles[j].velocity - particles[i].velocity)
                    / (particles[i].density + particles[j].density)
                    * lap_viscocity(
                        particles[i].position - particles[j].position,
                        PARTICLE_RADIUS,
                    );
            if !acceleration.is_nan() {
                particles[i].acceleration += acceleration;
            }
        }

        // gravity
        particles[i].acceleration += Vec2::NEG_Y * GRAVITY_ACCELERATION;
    }
}

fn compute_position_and_velocity(particles: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        // forward euler method
        particles[i].velocity += particles[i].acceleration * TIME_STEP;
        particles[i].position += particles[i].velocity * TIME_STEP;

        // bounds
        if particles[i].position.x < 0.0 {
            particles[i].position.x = 0.0;
            particles[i].velocity.x *= VELOCITY_DUMPING;
        } else if RANGE_X < particles[i].position.x {
            particles[i].position.x = RANGE_X;
            particles[i].velocity.x *= VELOCITY_DUMPING;
        }
        if particles[i].position.y < 0.0 {
            particles[i].position.y = 0.0;
            particles[i].velocity.y *= VELOCITY_DUMPING;
        } else if RANGE_Y < particles[i].position.y {
            particles[i].position.y = RANGE_Y;
            particles[i].velocity.y *= VELOCITY_DUMPING;
        }
    }
}

// renderer

fn draw(particles: &[Particle]) {
    let mut text = String::new();
    let mut amount = [[0; STDOUT_X]; STDOUT_Y];

    for i in 0..PARTICLE_SIZE {
        let x = (particles[i].position.x / RANGE_X * STDOUT_X as f32).floor() as i32;
        let y = (particles[i].position.y / RANGE_Y * STDOUT_Y as f32).floor() as i32;
        if 0 <= x && x < STDOUT_X as i32 && 0 <= y && y < STDOUT_Y as i32 {
            amount[y as usize][x as usize] += 1;
        }
    }

    for y in (0..STDOUT_Y).rev() {
        for x in 0..STDOUT_X {
            text.push(match amount[y][x] {
                a if a < 1 => ' ',
                a if a < 2 => '.',
                a if a < 3 => '-',
                a if a < 4 => '=',
                a if a < 5 => '*',
                a if a < 6 => '%',
                a if a < 7 => '$',
                _ => '#',
            });
        }
        text.push('\n');
    }

    print!("\x1b[2J\x1b[1;1H{}", text);
}

// main routine

fn main() {
    let mut rng = rand::thread_rng();

    // spawn particle
    let mut particles = [Particle::default(); PARTICLE_SIZE];
    for i in 0..PARTICLE_SIZE {
        particles[i].position =
            Vec2::new(rng.gen_range(0.0..=RANGE_X), rng.gen_range(0.0..=RANGE_Y));
    }

    loop {
        for _ in 0..ITERATE {
            compute_density_and_pressure(&mut particles);
            compute_acceleration(&mut particles);
            compute_position_and_velocity(&mut particles);
        }

        draw(&particles);

        std::thread::sleep(std::time::Duration::from_secs_f32(
            TIME_STEP * ITERATE as f32,
        ));
    }
}
