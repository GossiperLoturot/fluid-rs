use glam::*;

struct Config {
    dt: f32,
    iterations: i32,
    particle_size: i32,
    grid_res: i32,
    gravity: Vec2,
    rest_density: f32,
    dynamic_viscosity: f32,
    eos_stiffness: f32,
    eos_power: f32,
    mouse_radius: f32,
    boundary_clip: f32,
    boundary_damp: f32,
    console_size: IVec2,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dt: 0.032,
            iterations: (1.0 / 0.032) as i32,
            particle_size: 4096,
            grid_res: 64,
            gravity: Vec2::new(0.0, 0.3),
            rest_density: 4.0,
            dynamic_viscosity: 0.1,
            eos_stiffness: 10.0,
            eos_power: 4.0,
            mouse_radius: 10.0,
            boundary_clip: 1.0,
            boundary_damp: 3.0,
            console_size: IVec2::new(80, 40),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Particle {
    pos: Vec2,
    vel: Vec2,
    affine_momentum: Mat2,
    mass: f32,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Cell {
    vel: Vec2,
    mass: f32,
}

struct Simulation {
    config: Config,
    particles: Vec<Particle>,
    grid: Vec<Cell>,
    debug_elapseds: Vec<(&'static str, std::time::Duration)>,
}

impl Simulation {
    fn new(config: Config) -> Self {
        let particles = vec![Particle::default(); config.particle_size as usize];

        let grid_size = config.grid_res * config.grid_res;
        let grid = vec![Cell::default(); grid_size as usize];

        Self {
            config,
            particles,
            grid,
            debug_elapseds: Vec::new(),
        }
    }

    fn initialize_particles(&mut self) {
        use rand::Rng;

        let mut rng = rand::rng();

        for p in self.particles.iter_mut() {
            let x = rng.random_range(16.0..=48.0);
            let y = rng.random_range(16.0..=48.0);
            p.pos = Vec2::new(x, y);
            p.vel = Vec2::ZERO;
            p.affine_momentum = Mat2::ZERO;
            p.mass = 1.0;
        }
    }

    fn step(&mut self, mouse_pos: &Option<Vec2>) {
        for _ in 0..self.config.iterations {
            self.debug_elapseds.clear();

            let instance = std::time::Instant::now();
            self.clear_grid();
            self.debug_elapseds.push(("clear", instance.elapsed()));

            let instance = std::time::Instant::now();
            self.p2g_1();
            self.debug_elapseds.push(("p2g 1", instance.elapsed()));

            let instance = std::time::Instant::now();
            self.p2g_2();
            self.debug_elapseds.push(("p2g 2", instance.elapsed()));

            let instance = std::time::Instant::now();
            self.update_grid();
            self.debug_elapseds.push(("update", instance.elapsed()));

            let instance = std::time::Instant::now();
            self.g2p(mouse_pos);
            self.debug_elapseds.push(("g2p", instance.elapsed()));
        }
    }

    fn clear_grid(&mut self) {
        for cell in self.grid.iter_mut() {
            cell.vel = Vec2::ZERO;
            cell.mass = 0.0;
        }
    }

    fn p2g_1(&mut self) {
        let grid_res = self.config.grid_res;

        for p in self.particles.iter() {
            let cell_idx = p.pos.floor().as_ivec2();
            let cell_diff = p.pos - cell_idx.as_vec2() - Vec2::splat(0.5);
            let w = Self::quadratic_weights(cell_diff);

            for gy in 0..3 {
                for gx in 0..3 {
                    let weight = w[gx as usize].x * w[gy as usize].y;
                    let cell_pos = cell_idx + IVec2::new(gx - 1, gy - 1);
                    let cell_dist = cell_pos.as_vec2() - p.pos + Vec2::splat(0.5);

                    let q = p.affine_momentum * cell_dist;
                    let mass_contrib = weight * p.mass;

                    let grid_idx = (cell_pos.y * grid_res + cell_pos.x) as usize;
                    let cell = &mut self.grid[grid_idx];
                    cell.mass += mass_contrib;
                    cell.vel += mass_contrib * (p.vel + q);
                }
            }
        }
    }

    fn p2g_2(&mut self) {
        let grid_res = self.config.grid_res;
        let rest_density = self.config.rest_density;
        let eos_stiffness = self.config.eos_stiffness;
        let eos_power = self.config.eos_power;
        let dynamic_viscosity = self.config.dynamic_viscosity;
        let dt = self.config.dt;

        for p in self.particles.iter() {
            let cell_idx = p.pos.floor().as_ivec2();
            let cell_diff = p.pos - cell_idx.as_vec2() - Vec2::splat(0.5);
            let w = Self::quadratic_weights(cell_diff);

            let mut density = 0.0;
            for gy in 0..3 {
                for gx in 0..3 {
                    let weight = w[gx as usize].x * w[gy as usize].y;
                    let cell_pos = cell_idx + IVec2::new(gx - 1, gy - 1);

                    let grid_idx = (cell_pos.y * grid_res + cell_pos.x) as usize;
                    density += self.grid[grid_idx].mass * weight;
                }
            }
            let volume = p.mass / density;
            let pressure = f32::max(
                -0.1,
                eos_stiffness * ((density / rest_density).powf(eos_power) - 1.0),
            );

            let mut strain = p.affine_momentum;
            let trace = strain.y_axis.x + strain.x_axis.y;
            strain.y_axis.x = trace;
            strain.x_axis.y = trace;

            let viscosity_term = dynamic_viscosity * strain;
            let stress = -pressure * Mat2::IDENTITY + viscosity_term;
            let eg_16_term_0 = -4.0 * volume * stress * dt;

            for gy in 0..3 {
                for gx in 0..3 {
                    let weight = w[gx as usize].x * w[gy as usize].y;
                    let cell_pos = cell_idx + IVec2::new(gx - 1, gy - 1);
                    let cell_dist = cell_pos.as_vec2() - p.pos + Vec2::splat(0.5);

                    let grid_idx = (cell_pos.y * grid_res + cell_pos.x) as usize;
                    let cell = &mut self.grid[grid_idx];
                    cell.vel += weight * eg_16_term_0 * cell_dist;
                }
            }
        }
    }

    fn update_grid(&mut self) {
        let grid_res = self.config.grid_res;
        let dt = self.config.dt;
        let gravity = self.config.gravity;

        let boundary = self.config.boundary_clip.ceil() as i32;

        for i in 0..grid_res * grid_res {
            let cell = &mut self.grid[i as usize];

            if cell.mass > 0.0 {
                cell.vel /= cell.mass;
                cell.vel += dt * gravity;

                let x = i % grid_res;
                if x < boundary || x > grid_res - 1 - boundary {
                    cell.vel.x = 0.0;
                }

                let y = i / grid_res;
                if y < boundary || y > grid_res - 1 - boundary {
                    cell.vel.y = 0.0;
                }
            }
        }
    }

    fn g2p(&mut self, mouse_pos: &Option<Vec2>) {
        let grid_res = self.config.grid_res;
        let dt = self.config.dt;
        let mouse_radius = self.config.mouse_radius;
        let boundary_clip = self.config.boundary_clip;
        let boundary_damp = self.config.boundary_damp;

        for p in self.particles.iter_mut() {
            p.vel = Vec2::ZERO;

            let cell_idx = p.pos.floor().as_ivec2();
            let cell_diff = p.pos - cell_idx.as_vec2() - Vec2::splat(0.5);
            let w = Self::quadratic_weights(cell_diff);

            let mut b_mat = Mat2::ZERO;
            for gx in 0..3 {
                for gy in 0..3 {
                    let weight = w[gx as usize].x * w[gy as usize].y;

                    let cell_pos = cell_idx + IVec2::new(gx - 1, gy - 1);
                    let cell_dist = cell_pos.as_vec2() - p.pos + Vec2::splat(0.5);

                    let grid_idx = (cell_pos.y * grid_res + cell_pos.x) as usize;
                    let weighted_velocity = self.grid[grid_idx].vel * weight;

                    let term = Mat2::from_cols(
                        weighted_velocity * cell_dist.x,
                        weighted_velocity * cell_dist.y,
                    );
                    b_mat += term;
                    p.vel += weighted_velocity;
                }
            }

            p.affine_momentum = 4.0 * b_mat;
            p.pos += p.vel * dt;

            // mouse interaction

            if let Some(mouse) = mouse_pos {
                let dist = p.pos - mouse;
                if dist.length_squared() < mouse_radius.powi(2) {
                    p.vel += dist.normalize_or_zero();
                }
            }

            // boundary conditions

            p.pos = Vec2::clamp(
                p.pos,
                Vec2::splat(self.config.boundary_clip),
                IVec2::splat(grid_res - 1).as_vec2() - Vec2::splat(boundary_clip),
            );

            let next_pos = p.pos + p.vel;
            let wall_min = self.config.boundary_damp;
            let wall_max = (grid_res - 1) as f32 - boundary_damp;

            if next_pos.x < wall_min {
                p.vel.x += wall_min - next_pos.x;
            }
            if next_pos.x > wall_max {
                p.vel.x += wall_max - next_pos.x;
            }
            if next_pos.y < wall_min {
                p.vel.y += wall_min - next_pos.y;
            }
            if next_pos.y > wall_max {
                p.vel.y += wall_max - next_pos.y;
            }
        }
    }

    fn quadratic_weights(cell_diff: Vec2) -> [Vec2; 3] {
        [
            0.5 * (0.5 - cell_diff).powf(2.0),
            0.75 - cell_diff.powf(2.0),
            0.5 * (0.5 + cell_diff).powf(2.0),
        ]
    }
}

#[derive(Clone, Copy)]
enum Event {
    Quit,
    Drag(u16, u16),
}

fn setup_terminal(out: &mut impl std::io::Write) -> std::io::Result<()> {
    use crossterm::{cursor, event, terminal, ExecutableCommand};

    terminal::enable_raw_mode()?;
    out.execute(terminal::EnterAlternateScreen)?;
    out.execute(cursor::Hide)?;
    out.execute(event::EnableMouseCapture)?;
    Ok(())
}

fn restore_terminal(out: &mut impl std::io::Write) -> std::io::Result<()> {
    use crossterm::{cursor, event, terminal, ExecutableCommand};

    out.execute(event::DisableMouseCapture)?;
    out.execute(cursor::Show)?;
    out.execute(terminal::LeaveAlternateScreen)?;
    terminal::disable_raw_mode()?;
    Ok(())
}

fn event_handler(tx: crossbeam_channel::Sender<Event>) {
    use crossterm::event;

    loop {
        if let Ok(event) = event::read() {
            match event {
                event::Event::Key(key_event) => {
                    if key_event.code == event::KeyCode::Char('q') {
                        let _ = tx.send(Event::Quit);
                    }
                }
                event::Event::Mouse(mouse_event) => {
                    if matches!(mouse_event.kind, event::MouseEventKind::Down(_)) {
                        let _ = tx.try_send(Event::Drag(mouse_event.column, mouse_event.row));
                    }
                    if matches!(mouse_event.kind, event::MouseEventKind::Drag(_)) {
                        let _ = tx.try_send(Event::Drag(mouse_event.column, mouse_event.row));
                    }
                }
                _ => {}
            }
        }
    }
}

fn draw(out: &mut impl std::io::Write, sim: &Simulation) -> std::io::Result<()> {
    use crossterm::ExecutableCommand;

    let grid_res = sim.config.grid_res as f32;
    let console_size = sim.config.console_size;

    let bin_count_size = console_size.x * console_size.y;
    let mut bin_counts = vec![0; bin_count_size as usize];

    for p in sim.particles.iter() {
        let screen_pos = (p.pos / Vec2::splat(grid_res) * console_size.as_vec2()).as_ivec2();
        if screen_pos.cmpge(IVec2::ZERO).all() && screen_pos.cmplt(console_size).all() {
            let idx = screen_pos.y * console_size.x + screen_pos.x;
            bin_counts[idx as usize] += 1;
        }
    }

    for y in 0..console_size.y {
        out.execute(crossterm::cursor::MoveTo(0, y as u16))?;
        for x in 0..console_size.x {
            let idx = y * console_size.x + x;
            let bin_count = bin_counts[idx as usize];
            let ch = match bin_count {
                n if n < 1 => b' ',
                n if n < 2 => b'.',
                n if n < 3 => b'-',
                n if n < 4 => b'=',
                n if n < 5 => b'*',
                n if n < 6 => b'%',
                n if n < 7 => b'$',
                _ => b'#',
            };
            out.write_all(&[ch])?;
        }
    }

    for (i, (label, elapsed)) in sim.debug_elapseds.iter().enumerate() {
        let y = (console_size.y + i as i32) as u16;
        out.execute(crossterm::cursor::MoveTo(0, y))?;
        write!(out, "{}: {:?}", label, elapsed)?;
        out.execute(crossterm::terminal::Clear(
            crossterm::terminal::ClearType::FromCursorDown,
        ))?;
    }

    Ok(())
}

fn main() -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::io::stdout());
    setup_terminal(&mut out)?;

    let (tx, rx) = crossbeam_channel::bounded::<Event>(1);
    std::thread::spawn(|| event_handler(tx));

    let config = Config::default();
    let mut sim = Simulation::new(config);

    sim.initialize_particles();

    let console_size = sim.config.console_size;
    let grid_res = sim.config.grid_res;
    let time = std::time::Duration::from_secs_f32(sim.config.dt);
    loop {
        let mut mouse_pos = None;

        match rx.try_recv() {
            Ok(event) => match event {
                Event::Quit => break,
                Event::Drag(x, y) => {
                    let x = x as f32 / console_size.x as f32 * grid_res as f32;
                    let y = y as f32 / console_size.y as f32 * grid_res as f32;
                    mouse_pos = Some(Vec2::new(x, y));
                }
            },
            Err(crossbeam_channel::TryRecvError::Empty) => {}
            Err(crossbeam_channel::TryRecvError::Disconnected) => break,
        }

        draw(&mut out, &sim)?;

        sim.step(&mouse_pos);

        std::thread::sleep(time);
    }

    restore_terminal(&mut out)?;

    Ok(())
}
