use glam::*;

const PARTICLE_RADIUS: f32 = 0.012;
const PARTICLE_MASS: f32 = 0.0002;
const VISCOSITY: f32 = 0.1;
const PRESSURE_STIFFNESS: f32 = 200.0;
const REST_DENSITY: f32 = 1000.0;

struct Particle {
    position: Vec2,
    velosity: Vec2,
    density: f32,
    pressure: f32,
    force: Vec2,
}

fn compute_density(read: &[Particle], write: &mut [Particle]) {
    for self_particle in write {
        self_particle.density = 0.0;

        for neighbor_particle in read {
            let distance = neighbor_particle.position.distance(self_particle.position);

            self_particle.density += PARTICLE_MASS
                * if distance <= PARTICLE_RADIUS {
                    (PARTICLE_RADIUS.powi(2) - distance.powi(2)).powi(3)
                } else {
                    0.0
                };
        }
    }
}

fn compute_pressure(read: &[Particle], write: &mut [Particle]) {
    for self_particle in write {
        self_particle.pressure =
            PRESSURE_STIFFNESS * ((self_particle.density / REST_DENSITY).powi(7) - 1.0).max(0.0);
    }
}

fn compute_force(read: &[Particle], write: &mut [Particle]) {
    for self_particle in write {
        self_particle.force = Vec2::ZERO;

        for neighbor_particle in read {
            let diff = neighbor_particle.position - self_particle.position;
            let distance = diff.length();

            // viscosity force
            self_particle.force +=
                VISCOSITY * PARTICLE_MASS * (neighbor_particle.velosity - self_particle.velosity)
                    / neighbor_particle.density
                    * if distance <= PARTICLE_RADIUS {
                        20.0 / (3.0 * std::f32::consts::PI * PARTICLE_RADIUS.powi(5))
                            * (PARTICLE_RADIUS - distance)
                    } else {
                        0.0
                    };

            // pressure force
            self_particle.force += -1.0 / self_particle.pressure
                * PARTICLE_MASS
                * (neighbor_particle.pressure - self_particle.pressure)
                / (2.0 * neighbor_particle.density)
                * if distance <= PARTICLE_RADIUS {
                    -30.0 / (std::f32::consts::PI * PARTICLE_RADIUS.powi(5))
                        * (PARTICLE_RADIUS - distance).powi(2)
                        * diff.normalize()
                } else {
                    Vec2::ZERO
                };
        }
    }
}

fn main() {
    println!("Hello, world!");
}
