use glam::*;

const PARTICLE_SIZE: usize = 1024;
const PARTICLE_RADIUS: f32 = 0.012;
const PARTICLE_MASS: f32 = 0.0002;
const VISCOSITY: f32 = 0.1;
const PRESSURE_STIFFNESS: f32 = 200.0;
const REST_DENSITY: f32 = 1000.0;
const TIME_STEP: f32 = 0.01666;

#[derive(Debug, Default, Clone)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
    density: f32,
    pressure: f32,
    force: Vec2,
}

fn compute_density(read_buffer: &[Particle], write_buffer: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        write_buffer[i].density = 0.0;

        for j in 0..PARTICLE_SIZE {
            if i == j {
                continue;
            }

            let distance = read_buffer[i].position.distance(read_buffer[j].position);

            write_buffer[i].density += PARTICLE_MASS
                * if distance <= PARTICLE_RADIUS {
                    (PARTICLE_RADIUS.powi(2) - distance.powi(2)).powi(3)
                } else {
                    0.0
                };
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

            let distance = read_buffer[i].position.distance(read_buffer[j].position);

            // viscosity force
            write_buffer[i].force +=
                VISCOSITY * PARTICLE_MASS * (read_buffer[j].velocity - read_buffer[i].velocity)
                    / read_buffer[j].density
                    * if distance <= PARTICLE_RADIUS {
                        20.0 / (3.0 * std::f32::consts::PI * PARTICLE_RADIUS.powi(5))
                            * (PARTICLE_RADIUS - distance)
                    } else {
                        0.0
                    };

            // pressure force
            write_buffer[i].force += -1.0 / read_buffer[i].density
                * PARTICLE_MASS
                * (read_buffer[j].pressure - read_buffer[i].pressure)
                / (2.0 * read_buffer[j].density)
                * if distance <= PARTICLE_RADIUS {
                    -30.0 / (std::f32::consts::PI * PARTICLE_RADIUS.powi(5))
                        * (PARTICLE_RADIUS - distance).powi(2)
                        * (read_buffer[j].position - read_buffer[i].position)
                        / distance
                } else {
                    Vec2::ZERO
                };
        }
    }
}

fn compute_position_and_velocity(read_buffer: &[Particle], write_buffer: &mut [Particle]) {
    for i in 0..PARTICLE_SIZE {
        let acceleration = read_buffer[i].force / read_buffer[i].density;
        let velocity = read_buffer[i].velocity + acceleration * TIME_STEP;
        let position = read_buffer[i].position + velocity * TIME_STEP;

        write_buffer[i].velocity = velocity;
        write_buffer[i].position = position;
    }
}

fn main() {
    let read_buffer = vec![Particle::default(); PARTICLE_SIZE];
    let mut write_buffer = vec![Particle::default(); PARTICLE_SIZE];

    loop {
        compute_density(&read_buffer, &mut write_buffer);
        compute_pressure(&read_buffer, &mut write_buffer);
        compute_force(&read_buffer, &mut write_buffer);
        compute_position_and_velocity(&read_buffer, &mut write_buffer);

        std::thread::sleep(std::time::Duration::from_secs_f32(TIME_STEP));
    }
}
