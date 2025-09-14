use glam::*;

const MUL_SIZE: i32 = 32;

struct Config {
    dt: f32,
    iterations: i32,
    grid_res: i32,
    gravity: Vec2,
    rest_density: f32,
    dynamic_viscosity: f32,
    eos_stiffness: f32,
    eos_power: f32,
    mouse_radius: f32,
    boundary_clip: Vec4,
    boundary_damp_dist: f32,
    viewport_size: Vec2,
    console_size: IVec2,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dt: 0.032,
            iterations: (1.0 / 0.032) as i32,
            grid_res: 64,
            gravity: Vec2::new(0.0, 0.3),
            rest_density: 4.0,
            dynamic_viscosity: 0.1,
            eos_stiffness: 10.0,
            eos_power: 4.0,
            mouse_radius: 10.0,
            boundary_clip: Vec4::new(0.0, 0.0, 64.0, 64.0),
            boundary_damp_dist: 3.0,
            viewport_size: Vec2::new(64.0, 64.0),
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
    active_mul: Vec<IVec2>,
    particles_mul: ahash::AHashMap<IVec2, Vec<Particle>>,
    grid_mul: ahash::AHashMap<IVec2, Vec<Cell>>,
}

impl Simulation {
    fn new(config: Config) -> Self {
        Self {
            config,
            active_mul: Vec::new(),
            particles_mul: ahash::AHashMap::new(),
            grid_mul: ahash::AHashMap::new(),
        }
    }

    fn initialize_particles(&mut self) {
        use rand::Rng;

        let mut rng = rand::rng();

        self.active_mul.push(IVec2::new(0, 0));
        self.active_mul.push(IVec2::new(1, 0));
        self.active_mul.push(IVec2::new(0, 1));
        self.active_mul.push(IVec2::new(1, 1));

        for _ in 0..4096 {
            let x = rng.random_range(16.0..=48.0);
            let y = rng.random_range(16.0..=48.0);

            let p = Particle {
                pos: Vec2::new(x, y),
                vel: Vec2::ZERO,
                affine_momentum: Mat2::ZERO,
                mass: 1.0,
            };

            let key_x = x.div_euclid(MUL_SIZE as f32) as i32;
            let key_y = y.div_euclid(MUL_SIZE as f32) as i32;
            self.particles_mul
                .entry(IVec2::new(key_x, key_y))
                .or_default()
                .push(p);
        }
    }

    fn step(&mut self, mouse_pos: &Option<Vec2>) {
        for _ in 0..self.config.iterations {
            self.clear_grid();
            self.p2g_1();
            self.p2g_2();
            self.update_grid();
            self.g2p(mouse_pos);
        }
    }

    fn clear_grid(&mut self) {
        for (_, grid) in self.grid_mul.iter_mut() {
            for cell in grid.iter_mut() {
                cell.vel = Vec2::ZERO;
                cell.mass = 0.0;
            }
        }
    }

    fn p2g_1(&mut self) {
        let grid_res = self.config.grid_res;

        for (_, particles) in self.particles_mul.iter() {
            for p in particles.iter() {
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

                        // TODO: optimize
                        let key = cell_pos.div_euclid(IVec2::splat(MUL_SIZE));
                        let grid = self
                            .grid_mul
                            .entry(key)
                            .or_insert_with(|| vec![Cell::default(); grid_res.pow(2) as usize]);
                        let local_pos = cell_pos.rem_euclid(IVec2::splat(MUL_SIZE));
                        let grid_idx = (local_pos.y * grid_res + local_pos.x) as usize;

                        let cell = &mut grid[grid_idx];
                        cell.mass += mass_contrib;
                        cell.vel += mass_contrib * (p.vel + q);
                    }
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

        for (_, particles) in self.particles_mul.iter() {
            for p in particles.iter() {
                let cell_idx = p.pos.floor().as_ivec2();
                let cell_diff = p.pos - cell_idx.as_vec2() - Vec2::splat(0.5);
                let w = Self::quadratic_weights(cell_diff);

                let mut density = 0.0;
                for gy in 0..3 {
                    for gx in 0..3 {
                        let weight = w[gx as usize].x * w[gy as usize].y;
                        let cell_pos = cell_idx + IVec2::new(gx - 1, gy - 1);

                        // TODO: optimize
                        let key = cell_pos.div_euclid(IVec2::splat(MUL_SIZE));
                        let grid = self
                            .grid_mul
                            .entry(key)
                            .or_insert_with(|| vec![Cell::default(); grid_res.pow(2) as usize]);
                        let local_pos = cell_pos.rem_euclid(IVec2::splat(MUL_SIZE));
                        let grid_idx = (local_pos.y * grid_res + local_pos.x) as usize;

                        density += grid[grid_idx].mass * weight;
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

                        // TODO: optimize
                        let key = cell_pos.div_euclid(IVec2::splat(MUL_SIZE));
                        let grid = self
                            .grid_mul
                            .entry(key)
                            .or_insert_with(|| vec![Cell::default(); grid_res.pow(2) as usize]);
                        let local_pos = cell_pos.rem_euclid(IVec2::splat(MUL_SIZE));
                        let grid_idx = (local_pos.y * grid_res + local_pos.x) as usize;

                        let cell = &mut grid[grid_idx];
                        cell.vel += weight * eg_16_term_0 * cell_dist;
                    }
                }
            }
        }
    }

    fn update_grid(&mut self) {
        let grid_res = self.config.grid_res;
        let dt = self.config.dt;
        let gravity = self.config.gravity;

        for (_, grid) in self.grid_mul.iter_mut() {
            for i in 0..grid_res * grid_res {
                let cell = &mut grid[i as usize];

                if cell.mass > 0.0 {
                    cell.vel /= cell.mass;
                    cell.vel += dt * gravity;
                }
            }
        }
    }

    fn g2p(&mut self, mouse_pos: &Option<Vec2>) {
        let grid_res = self.config.grid_res;
        let dt = self.config.dt;
        let mouse_radius = self.config.mouse_radius;
        let boundary_clip = self.config.boundary_clip;
        let boundary_damp_dist = self.config.boundary_damp_dist;

        let mut move_buf = vec![];

        for (key, particles) in self.particles_mul.iter_mut() {
            for (i, p) in particles.iter_mut().enumerate() {
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

                        // TODO: optimize
                        let key = cell_pos.div_euclid(IVec2::splat(MUL_SIZE));
                        let grid = self
                            .grid_mul
                            .entry(key)
                            .or_insert_with(|| vec![Cell::default(); grid_res.pow(2) as usize]);
                        let local_pos = cell_pos.rem_euclid(IVec2::splat(MUL_SIZE));
                        let grid_idx = (local_pos.y * grid_res + local_pos.x) as usize;

                        let weighted_velocity = grid[grid_idx].vel * weight;

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

                p.pos = Vec2::clamp(p.pos, boundary_clip.xy(), boundary_clip.zw());

                let next_pos = p.pos + p.vel;
                let wall_min = boundary_clip.xy() + Vec2::splat(boundary_damp_dist);
                let wall_max = boundary_clip.zw() - Vec2::splat(boundary_damp_dist);

                if next_pos.x < wall_min.x {
                    p.vel.x += wall_min.x - next_pos.x;
                }
                if next_pos.x > wall_max.x {
                    p.vel.x += wall_max.x - next_pos.x;
                }
                if next_pos.y < wall_min.y {
                    p.vel.y += wall_min.y - next_pos.y;
                }
                if next_pos.y > wall_max.y {
                    p.vel.y += wall_max.y - next_pos.y;
                }

                let to_key = p.pos.rem_euclid(Vec2::splat(MUL_SIZE as f32)).as_ivec2();
                if to_key != *key {
                    move_buf.push((*key, i, to_key));
                }
            }
        }

        for (from_key, i, to_key) in move_buf.into_iter().rev() {
            let from_particles = self.particles_mul.get_mut(&from_key).unwrap();
            let p = from_particles.swap_remove(i);
            let to_partucles = self.particles_mul.entry(to_key).or_default();
            to_partucles.push(p);
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

    let viewport_size = sim.config.viewport_size;
    let console_size = sim.config.console_size;

    let bin_count_size = console_size.x * console_size.y;
    let mut bin_counts = vec![0; bin_count_size as usize];

    for (_, particles) in sim.particles_mul.iter() {
        for p in particles.iter() {
            let screen_pos = (p.pos / viewport_size * console_size.as_vec2()).as_ivec2();

            if screen_pos.cmplt(IVec2::ZERO).any() || screen_pos.cmpge(console_size).any() {
                continue;
            }

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
