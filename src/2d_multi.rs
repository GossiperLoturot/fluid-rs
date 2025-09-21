use glam::*;

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
    boundary_clip: (Vec2, Vec2),
    boundary_damp_dist: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dt: 0.032,
            iterations: (1.0 / 0.032) as i32,
            grid_res: 32,
            gravity: Vec2::new(0.0, 0.3),
            rest_density: 4.0,
            dynamic_viscosity: 0.1,
            eos_stiffness: 10.0,
            eos_power: 4.0,
            mouse_radius: 10.0,
            boundary_clip: (Vec2::new(0.0, 0.0), Vec2::new(64.0, 64.0)),
            boundary_damp_dist: 3.0,
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
    is_computed: bool,
}

struct Simulation {
    config: Config,
    particles_mul: ahash::AHashMap<IVec2, Vec<Particle>>,
    grid_mul: Vec<Cell>,
    grid_size: IVec2,
    sparse_grid: Vec<i32>,
    swap_mul: Vec<Vec<Particle>>,
    swap_size: IVec2,
    p_rect: (IVec2, IVec2),
    a_rect: (IVec2, IVec2),
    debug_elapseds: Vec<(&'static str, std::time::Duration)>,
}

impl Simulation {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            particles_mul: ahash::AHashMap::new(),
            grid_mul: Vec::new(),
            grid_size: IVec2::ZERO,
            sparse_grid: Vec::new(),
            swap_mul: Vec::new(),
            swap_size: IVec2::ZERO,
            p_rect: (IVec2::ZERO, IVec2::ZERO),
            a_rect: (IVec2::ZERO, IVec2::ZERO),
            debug_elapseds: Vec::new(),
        }
    }

    pub fn set_rect(&mut self, min: Vec2, max: Vec2) {
        let min_key = key_from_pos(min, &self.config);
        let max_key = key_from_pos(max, &self.config) + IVec2::ONE;
        self.a_rect = (min_key, max_key);

        let min_key = min_key - IVec2::ONE;
        let max_key = max_key + IVec2::ONE;
        self.p_rect = (min_key, max_key);

        // for particles
        for k in grid_search(&self.p_rect.0, &self.p_rect.1) {
            let _ = self.particles_mul.entry(k).or_default();
        }

        // for grid
        self.grid_size = (self.p_rect.1 - self.p_rect.0) * self.config.grid_res;
        let size = self.grid_size.x * self.grid_size.y;
        self.grid_mul = vec![Default::default(); size as usize];

        // for move
        self.swap_size = self.p_rect.1 - self.p_rect.0;
        let size = self.swap_size.x * self.swap_size.y;
        self.swap_mul = vec![Default::default(); size as usize];
    }

    pub fn add_particle(&mut self, p: Particle) {
        let k = key_from_pos(p.pos, &self.config);
        let particles = self.particles_mul.entry(k).or_default();
        particles.push(p);
    }

    pub fn step(&mut self, mouse_pos: &Option<Vec2>) {
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
        for index in self.sparse_grid.iter() {
            let cell = self.grid_mul.get_mut(*index as usize).unwrap();

            cell.vel = Vec2::ZERO;
            cell.mass = 0.0;
            cell.is_computed = false;
        }

        self.sparse_grid.clear();
    }

    fn p2g_1(&mut self) {
        for k in grid_search(&self.p_rect.0, &self.p_rect.1) {
            let particles = self.particles_mul.get(&k).unwrap();

            for p in particles.iter() {
                let cell_pos = p.pos.floor().as_ivec2();
                let cell_diff = p.pos - (cell_pos.as_vec2() + Vec2::splat(0.5));
                let ws = quadratic_weights(cell_diff);

                for n in grid_search(&IVec2::splat(0), &IVec2::splat(3)) {
                    let cell_pos_n = cell_pos + n - IVec2::ONE;
                    let cell_diff_n = p.pos - (cell_pos_n.as_vec2() + Vec2::splat(0.5));
                    let w = ws[n.x as usize].x * ws[n.y as usize].y;

                    let q = p.affine_momentum * -cell_diff_n;
                    let mass_contrib = w * p.mass;

                    let c1 = cell_pos_n.cmplt(self.p_rect.0 * self.config.grid_res).any();
                    let c2 = cell_pos_n.cmpge(self.p_rect.1 * self.config.grid_res).any();
                    if !(c1 || c2) {
                        let index_xy = cell_pos_n - self.p_rect.0 * self.config.grid_res;
                        let index = index_xy.x + index_xy.y * self.grid_size.x;
                        let cell_n = self.grid_mul.get_mut(index as usize).unwrap();

                        cell_n.mass += mass_contrib;
                        cell_n.vel += mass_contrib * (p.vel + q);

                        self.sparse_grid.push(index);
                    }
                }
            }
        }
    }

    fn p2g_2(&mut self) {
        for k in grid_search(&self.p_rect.0, &self.p_rect.1) {
            let particles = self.particles_mul.get(&k).unwrap();

            for p in particles.iter() {
                let cell_pos = p.pos.floor().as_ivec2();
                let cell_diff = p.pos - (cell_pos.as_vec2() + Vec2::splat(0.5));
                let ws = quadratic_weights(cell_diff);

                let rest_density = self.config.rest_density;
                let eos_stiffness = self.config.eos_stiffness;
                let eos_power = self.config.eos_power;

                let mut density = 0.0;
                for n in grid_search(&IVec2::splat(0), &IVec2::splat(3)) {
                    let cell_pos_n = cell_pos + n - IVec2::ONE;
                    let w = ws[n.x as usize].x * ws[n.y as usize].y;

                    let c1 = cell_pos_n.cmplt(self.p_rect.0 * self.config.grid_res).any();
                    let c2 = cell_pos_n.cmpge(self.p_rect.1 * self.config.grid_res).any();
                    if !(c1 || c2) {
                        let index_xy = cell_pos_n - self.p_rect.0 * self.config.grid_res;
                        let index = index_xy.x + index_xy.y * self.grid_size.x;
                        let cell_n = self.grid_mul.get(index as usize).unwrap();

                        density += cell_n.mass * w;
                    }
                }
                let volume = p.mass / density;
                let pressure = f32::max(
                    -0.0,
                    eos_stiffness * ((density / rest_density).powf(eos_power) - 1.0),
                );

                let strain = p.affine_momentum + p.affine_momentum.transpose();
                let viscosity_term = self.config.dynamic_viscosity * strain;
                let stress = -pressure * Mat2::IDENTITY + viscosity_term;
                let eg_16_term_0 = -4.0 * volume * stress * self.config.dt;

                for n in grid_search(&IVec2::splat(0), &IVec2::splat(3)) {
                    let cell_pos_n = cell_pos + n - IVec2::ONE;
                    let cell_diff_n = p.pos - (cell_pos_n.as_vec2() + Vec2::splat(0.5));
                    let w = ws[n.x as usize].x * ws[n.y as usize].y;

                    let c1 = cell_pos_n.cmplt(self.p_rect.0 * self.config.grid_res).any();
                    let c2 = cell_pos_n.cmpge(self.p_rect.1 * self.config.grid_res).any();
                    if !(c1 || c2) {
                        let index_xy = cell_pos_n - self.p_rect.0 * self.config.grid_res;
                        let index = index_xy.x + index_xy.y * self.grid_size.x;
                        let cell_n = self.grid_mul.get_mut(index as usize).unwrap();

                        cell_n.vel += w * eg_16_term_0 * -cell_diff_n;
                    }
                }
            }
        }
    }

    fn update_grid(&mut self) {
        for index in self.sparse_grid.iter() {
            let cell = self.grid_mul.get_mut(*index as usize).unwrap();

            if !cell.is_computed && cell.mass > 0.0 {
                cell.vel /= cell.mass;
                cell.vel += self.config.dt * self.config.gravity;
                cell.is_computed = true;
            }
        }
    }

    fn g2p(&mut self, mouse_pos: &Option<Vec2>) {
        let mut move_buf = vec![];
        for k in grid_search(&self.a_rect.0, &self.a_rect.1) {
            let particles = self.particles_mul.get_mut(&k).unwrap();

            move_buf.clear();
            for (i, p) in particles.iter_mut().enumerate() {
                p.vel = Vec2::ZERO;

                let cell_pos = p.pos.floor().as_ivec2();
                let cell_diff = p.pos - (cell_pos.as_vec2() + Vec2::splat(0.5));
                let ws = quadratic_weights(cell_diff);

                let mut b_mat = Mat2::ZERO;
                for n in grid_search(&IVec2::splat(0), &IVec2::splat(3)) {
                    let cell_pos_n = cell_pos + n - IVec2::ONE;
                    let cell_diff_n = p.pos - (cell_pos_n.as_vec2() + Vec2::splat(0.5));
                    let w = ws[n.x as usize].x * ws[n.y as usize].y;

                    let c1 = cell_pos_n.cmplt(self.p_rect.0 * self.config.grid_res).any();
                    let c2 = cell_pos_n.cmpge(self.p_rect.1 * self.config.grid_res).any();
                    if !(c1 || c2) {
                        let index_xy = cell_pos_n - self.p_rect.0 * self.config.grid_res;
                        let index = index_xy.x + index_xy.y * self.grid_size.x;
                        let cell_n = self.grid_mul.get(index as usize).unwrap();

                        let weighted_velocity = cell_n.vel * w;
                        let term = Mat2::from_cols(
                            weighted_velocity * -cell_diff_n.x,
                            weighted_velocity * -cell_diff_n.y,
                        );
                        b_mat += term;
                        p.vel += weighted_velocity;
                    }
                }

                p.affine_momentum = 4.0 * b_mat;
                p.pos += p.vel * self.config.dt;

                // mouse interaction

                if let Some(mouse) = mouse_pos {
                    let dist = p.pos - mouse;
                    if dist.length_squared() < self.config.mouse_radius * self.config.mouse_radius {
                        p.vel += dist.normalize_or_zero();
                    }
                }

                // boundary conditions

                p.pos = Vec2::clamp(
                    p.pos,
                    self.config.boundary_clip.0,
                    self.config.boundary_clip.1,
                );

                let next_pos = p.pos + p.vel;
                let wall_min =
                    self.config.boundary_clip.0 + Vec2::splat(self.config.boundary_damp_dist);
                let wall_max =
                    self.config.boundary_clip.1 - Vec2::splat(self.config.boundary_damp_dist);

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

                // swap

                let new_k = key_from_pos(p.pos, &self.config);
                if new_k != k {
                    move_buf.push((i as i32, new_k));
                }
            }

            for (i, k) in move_buf.drain(..).rev() {
                let p = particles.swap_remove(i as usize);

                let c1 = k.cmplt(self.p_rect.0).any();
                let c2 = k.cmpge(self.p_rect.1).any();
                if !(c1 || c2) {
                    let index_xyz = k - self.p_rect.0;
                    let index = index_xyz.x + index_xyz.y * self.swap_size.x;
                    let r#move = self.swap_mul.get_mut(index as usize).unwrap();

                    r#move.push(p);
                }
            }
        }

        for k in grid_search(&self.p_rect.0, &self.p_rect.1) {
            let particles = self.particles_mul.get_mut(&k).unwrap();

            let index_xy = k - self.p_rect.0;
            let index = index_xy.x + index_xy.y * self.swap_size.x;
            let r#move = self.swap_mul.get_mut(index as usize).unwrap();

            particles.append(r#move);
        }
    }

    pub fn iter_particle(&self) -> impl Iterator<Item = &Particle> + '_ {
        grid_search(&self.a_rect.0, &self.a_rect.1)
            .filter_map(|key| self.particles_mul.get(&key))
            .flatten()
    }
}

fn quadratic_weights(cell_diff: Vec2) -> [Vec2; 3] {
    [
        0.5 * (0.5 - cell_diff) * (0.5 - cell_diff),
        0.75 - cell_diff * cell_diff,
        0.5 * (0.5 + cell_diff) * (0.5 + cell_diff),
    ]
}

fn key_from_pos(pos: Vec2, config: &Config) -> IVec2 {
    pos.div_euclid(Vec2::splat(config.grid_res as f32))
        .as_ivec2()
}

fn grid_search(min: &IVec2, max: &IVec2) -> impl Iterator<Item = IVec2> {
    (min.y..max.y)
        .flat_map(|y| (min.x..max.x).map(move |x| (x, y)))
        .map(|(x, y)| IVec2::new(x, y))
}

#[derive(Clone, Copy)]
enum Event {
    Quit,
    Drag(u16, u16),
}

fn setup_terminal(out: &mut impl std::io::Write) -> std::io::Result<()> {
    use crossterm::{ExecutableCommand, cursor, event, terminal};

    terminal::enable_raw_mode()?;
    out.execute(terminal::EnterAlternateScreen)?;
    out.execute(cursor::Hide)?;
    out.execute(event::EnableMouseCapture)?;
    Ok(())
}

fn restore_terminal(out: &mut impl std::io::Write) -> std::io::Result<()> {
    use crossterm::{ExecutableCommand, cursor, event, terminal};

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

fn draw(
    out: &mut impl std::io::Write,
    sim: &Simulation,
    viewport_size: Vec2,
    console_size: IVec2,
) -> std::io::Result<()> {
    use crossterm::ExecutableCommand;

    let bin_count_size = console_size.x * console_size.y;
    let mut bin_counts = vec![0; bin_count_size as usize];

    for p in sim.iter_particle() {
        let console_xy = (p.pos / viewport_size * console_size.as_vec2()).as_ivec2();

        if console_xy.cmplt(IVec2::ZERO).any() || console_xy.cmpge(console_size).any() {
            continue;
        }

        let index = console_xy.y * console_size.x + console_xy.x;
        bin_counts[index as usize] += 1;
    }

    for y in 0..console_size.y {
        out.execute(crossterm::cursor::MoveTo(0, y as u16))?;
        for x in 0..console_size.x {
            let index = y * console_size.x + x;
            let bin_count = bin_counts[index as usize];
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

    let mut rng = rand::rng();
    for _ in 0..4096 {
        sim.add_particle(Particle {
            pos: Vec2::new(
                rand::Rng::random_range(&mut rng, 16.0..=48.0),
                rand::Rng::random_range(&mut rng, 16.0..=48.0),
            ),
            vel: Vec2::ZERO,
            affine_momentum: Mat2::ZERO,
            mass: 1.0,
        });
    }
    sim.set_rect(Vec2::new(0.0, 0.0), Vec2::new(64.0, 64.0));

    let viewport_size = Vec2::new(64.0, 64.0);
    let console_size = IVec2::new(80, 40);
    let time = std::time::Duration::from_secs_f32(sim.config.dt);
    loop {
        let mut mouse_pos = None;

        match rx.try_recv() {
            Ok(event) => match event {
                Event::Quit => break,
                Event::Drag(x, y) => {
                    let console_xy = Vec2::new(x as f32, y as f32);
                    let world_xy = console_xy / console_size.as_vec2() * viewport_size;
                    mouse_pos = Some(world_xy);
                }
            },
            Err(crossbeam_channel::TryRecvError::Empty) => {}
            Err(crossbeam_channel::TryRecvError::Disconnected) => break,
        }

        draw(&mut out, &sim, viewport_size, console_size)?;

        sim.step(&mouse_pos);

        std::thread::sleep(time);
    }

    restore_terminal(&mut out)?;

    Ok(())
}
