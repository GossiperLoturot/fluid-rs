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
    boundary_clip: Vec4,
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
            boundary_clip: Vec4::new(0.0, 0.0, 64.0, 64.0),
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
}

struct Simulation {
    config: Config,
    spatial_idx: ahash::AHashMap<IVec2, usize>,
    spatial_idx_rev: Vec<IVec2>,
    precompute_mul: Vec<usize>,
    passive_mul: Vec<usize>,
    active_mul: Vec<usize>,
    particles_mul: Vec<Vec<Particle>>,
    grid_mul: Vec<Vec<Cell>>,
    neighbors_mul: Vec<Vec<usize>>,
    debug_elapseds: Vec<(&'static str, std::time::Duration)>,
}

impl Simulation {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            spatial_idx: ahash::AHashMap::new(),
            spatial_idx_rev: Vec::new(),
            precompute_mul: Vec::new(),
            passive_mul: Vec::new(),
            active_mul: Vec::new(),
            particles_mul: Vec::new(),
            grid_mul: Vec::new(),
            neighbors_mul: Vec::new(),
            debug_elapseds: Vec::new(),
        }
    }

    pub fn set_update_rect(&mut self, min: Vec2, max: Vec2) {
        // pre-compute
        for id in self.precompute_mul.drain(..) {
            free_space(
                &self.spatial_idx,
                &self.spatial_idx_rev,
                &mut self.particles_mul,
                &mut self.grid_mul,
                &mut self.neighbors_mul,
                id,
            );
        }
        let min_key = key_from_pos(min, &self.config) - IVec2::ONE;
        let max_key = key_from_pos(max, &self.config) + IVec2::ONE;
        for y in min_key.y..=max_key.y {
            for x in min_key.x..=max_key.x {
                init_space(
                    &mut self.spatial_idx,
                    &mut self.spatial_idx_rev,
                    &mut self.particles_mul,
                    &mut self.grid_mul,
                    &mut self.neighbors_mul,
                    &IVec2::new(x, y),
                );

                let id = self.spatial_idx.get(&IVec2::new(x, y)).unwrap();
                self.precompute_mul.push(*id);
            }
        }
        for id in self.precompute_mul.iter() {
            bake_space(
                &self.spatial_idx,
                &self.spatial_idx_rev,
                &mut self.particles_mul,
                &mut self.grid_mul,
                &mut self.neighbors_mul,
                &self.config,
                *id,
            );
        }

        // passive
        self.passive_mul.clear();
        let min_key = key_from_pos(min, &self.config) - IVec2::ONE;
        let max_key = key_from_pos(max, &self.config) + IVec2::ONE;
        for y in min_key.y..=max_key.y {
            for x in min_key.x..=max_key.x {
                let id = self.spatial_idx.get(&IVec2::new(x, y)).unwrap();
                self.passive_mul.push(*id);
            }
        }

        // active
        self.active_mul.clear();
        let min_key = key_from_pos(min, &self.config);
        let max_key = key_from_pos(max, &self.config);
        for y in min_key.y..=max_key.y {
            for x in min_key.x..=max_key.x {
                let id = self.spatial_idx.get(&IVec2::new(x, y)).unwrap();
                self.active_mul.push(*id);
            }
        }
    }

    pub fn add_particle(&mut self, p: Particle) {
        let key = key_from_pos(p.pos, &self.config);

        init_space(
            &mut self.spatial_idx,
            &mut self.spatial_idx_rev,
            &mut self.particles_mul,
            &mut self.grid_mul,
            &mut self.neighbors_mul,
            &key,
        );

        let id = self.spatial_idx.get(&key).unwrap();
        let particles = self.particles_mul.get_mut(*id).unwrap();
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
        for id in self.passive_mul.iter() {
            let grid = self.grid_mul.get_mut(*id).unwrap();

            for cell in grid.iter_mut() {
                cell.vel = Vec2::ZERO;
                cell.mass = 0.0;
            }
        }
    }

    fn p2g_1(&mut self) {
        for id in self.passive_mul.iter() {
            let particles = self.particles_mul.get(*id).unwrap();

            for p in particles.iter() {
                let cell_pos = p.pos.floor().as_ivec2();
                let cell_diff = p.pos - (cell_pos.as_vec2() + Vec2::splat(0.5));
                let weights = quadratic_weights(cell_diff);

                for ny in 0..3 {
                    for nx in 0..3 {
                        let cell_pos_n = cell_pos + IVec2::new(nx - 1, ny - 1);
                        let cell_diff_n = p.pos - (cell_pos_n.as_vec2() + Vec2::splat(0.5));
                        let weight = weights[nx as usize].x * weights[ny as usize].y;

                        let q = p.affine_momentum * -cell_diff_n;
                        let mass_contrib = weight * p.mass;

                        if let Some(cell_n) = cell_mut_from_pos(
                            &self.spatial_idx,
                            &self.spatial_idx_rev,
                            &mut self.grid_mul,
                            &self.neighbors_mul,
                            &self.config,
                            *id,
                            &cell_pos_n,
                        ) {
                            cell_n.mass += mass_contrib;
                            cell_n.vel += mass_contrib * (p.vel + q);
                        }
                    }
                }
            }
        }
    }

    fn p2g_2(&mut self) {
        let rest_density = self.config.rest_density;
        let eos_stiffness = self.config.eos_stiffness;
        let eos_power = self.config.eos_power;
        let dynamic_viscosity = self.config.dynamic_viscosity;
        let dt = self.config.dt;

        for id in self.passive_mul.iter() {
            let particles = self.particles_mul.get(*id).unwrap();

            for p in particles.iter() {
                let cell_pos = p.pos.floor().as_ivec2();
                let cell_diff = p.pos - (cell_pos.as_vec2() + Vec2::splat(0.5));
                let weights = quadratic_weights(cell_diff);

                let mut density = 0.0;
                for ny in 0..3 {
                    for nx in 0..3 {
                        let cell_pos_n = cell_pos + IVec2::new(nx - 1, ny - 1);
                        let weight = weights[nx as usize].x * weights[ny as usize].y;

                        if let Some(cell_n) = cell_mut_from_pos(
                            &self.spatial_idx,
                            &self.spatial_idx_rev,
                            &mut self.grid_mul,
                            &self.neighbors_mul,
                            &self.config,
                            *id,
                            &cell_pos_n,
                        ) {
                            density += cell_n.mass * weight;
                        }
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

                for ny in 0..3 {
                    for nx in 0..3 {
                        let cell_pos_n = cell_pos + IVec2::new(nx - 1, ny - 1);
                        let cell_diff_n = p.pos - (cell_pos_n.as_vec2() + Vec2::splat(0.5));
                        let weight = weights[nx as usize].x * weights[ny as usize].y;

                        if let Some(cell_n) = cell_mut_from_pos(
                            &self.spatial_idx,
                            &self.spatial_idx_rev,
                            &mut self.grid_mul,
                            &self.neighbors_mul,
                            &self.config,
                            *id,
                            &cell_pos_n,
                        ) {
                            cell_n.vel += weight * eg_16_term_0 * -cell_diff_n;
                        }
                    }
                }
            }
        }
    }

    fn update_grid(&mut self) {
        let dt = self.config.dt;
        let gravity = self.config.gravity;

        for id in self.passive_mul.iter() {
            let grid = self.grid_mul.get_mut(*id).unwrap();

            for cell in grid.iter_mut() {
                if cell.mass > 0.0 {
                    cell.vel /= cell.mass;
                    cell.vel += dt * gravity;
                }
            }
        }
    }

    fn g2p(&mut self, mouse_pos: &Option<Vec2>) {
        let dt = self.config.dt;
        let mouse_radius = self.config.mouse_radius;
        let boundary_clip = self.config.boundary_clip;
        let boundary_damp_dist = self.config.boundary_damp_dist;

        let mut move_buf = vec![];
        for id in self.active_mul.iter() {
            let particles = self.particles_mul.get_mut(*id).unwrap();

            for mut p in particles.drain(..) {
                p.vel = Vec2::ZERO;

                let cell_pos = p.pos.floor().as_ivec2();
                let cell_diff = p.pos - (cell_pos.as_vec2() + Vec2::splat(0.5));
                let weights = quadratic_weights(cell_diff);

                let mut b_mat = Mat2::ZERO;
                for ny in 0..3 {
                    for nx in 0..3 {
                        let cell_pos_n = cell_pos + IVec2::new(nx - 1, ny - 1);
                        let cell_diff_n = p.pos - (cell_pos_n.as_vec2() + Vec2::splat(0.5));
                        let weight = weights[nx as usize].x * weights[ny as usize].y;

                        if let Some(cell_n) = cell_mut_from_pos(
                            &self.spatial_idx,
                            &self.spatial_idx_rev,
                            &mut self.grid_mul,
                            &self.neighbors_mul,
                            &self.config,
                            *id,
                            &cell_pos_n,
                        ) {
                            let weighted_velocity = cell_n.vel * weight;

                            let term = Mat2::from_cols(
                                weighted_velocity * -cell_diff_n.x,
                                weighted_velocity * -cell_diff_n.y,
                            );
                            b_mat += term;
                            p.vel += weighted_velocity;
                        }
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

                let new_id = 0; // TODO
                move_buf.push((new_id, p));
            }
        }

        for (id, p) in move_buf.into_iter() {
            let particles = self.particles_mul.get_mut(id).unwrap();
            particles.push(p);
        }
    }

    pub fn iter_particle(&self) -> impl Iterator<Item = &Particle> + '_ {
        self.active_mul
            .iter()
            .filter_map(|id| self.particles_mul.get(*id))
            .flatten()
    }
}

fn quadratic_weights(cell_diff: Vec2) -> [Vec2; 3] {
    [
        0.5 * (0.5 - cell_diff).powf(2.0),
        0.75 - cell_diff.powf(2.0),
        0.5 * (0.5 + cell_diff).powf(2.0),
    ]
}

fn key_from_pos(pos: Vec2, config: &Config) -> IVec2 {
    pos.div_euclid(Vec2::splat(config.grid_res as f32))
        .as_ivec2()
}

fn init_space(
    spatial_idx: &mut ahash::AHashMap<IVec2, usize>,
    spatial_idx_rev: &mut Vec<IVec2>,
    particles_mul: &mut Vec<Vec<Particle>>,
    grid_mul: &mut Vec<Vec<Cell>>,
    neighbors_mul: &mut Vec<Vec<usize>>,
    key: &IVec2,
) {
    if !spatial_idx.contains_key(&key) {
        let id = spatial_idx_rev.len();

        spatial_idx_rev.push(*key);
        particles_mul.push(Default::default());
        grid_mul.push(Default::default());
        neighbors_mul.push(Default::default());

        spatial_idx.insert(*key, id);
    }
}

fn bake_space(
    spatial_idx: &ahash::AHashMap<IVec2, usize>,
    spatial_idx_rev: &Vec<IVec2>,
    _particles_mul: &mut Vec<Vec<Particle>>,
    grid_mul: &mut Vec<Vec<Cell>>,
    neighbors_mul: &mut Vec<Vec<usize>>,
    config: &Config,
    id: usize,
) {
    // for grid
    let grid = grid_mul.get_mut(id).unwrap();
    *grid = vec![Default::default(); (config.grid_res * config.grid_res) as usize];

    // for neighbors
    let key = spatial_idx_rev.get(id).unwrap();
    let neighbors = neighbors_mul.get_mut(id).unwrap();
    *neighbors = vec![Default::default(); 9];
    for ny in 0..3 {
        for nx in 0..3 {
            let key = key + IVec2::new(nx as i32 - 1, ny as i32 - 1);
            let neighbor_id = 3 * ny + nx;
            neighbors[neighbor_id] = *spatial_idx.get(&key).unwrap_or(&usize::MAX);
        }
    }
}

fn free_space(
    _spatial_idx: &ahash::AHashMap<IVec2, usize>,
    _spatial_idx_rev: &Vec<IVec2>,
    _particles_mul: &mut Vec<Vec<Particle>>,
    grid_mul: &mut Vec<Vec<Cell>>,
    neighbors_mul: &mut Vec<Vec<usize>>,
    id: usize,
) {
    // for grid
    let grid = grid_mul.get_mut(id).unwrap();
    *grid = vec![];

    // for neighbors
    let neighbors = neighbors_mul.get_mut(id).unwrap();
    *neighbors = vec![];
}

fn cell_mut_from_pos<'a>(
    _spatial_idx: &'a ahash::AHashMap<IVec2, usize>,
    spatial_idx_rev: &'a Vec<IVec2>,
    grid_mul: &'a mut Vec<Vec<Cell>>,
    neighbors_mul: &'a Vec<Vec<usize>>,
    config: &Config,
    id: usize,
    pos: &'a IVec2,
) -> Option<&'a mut Cell> {
    let neighbors = neighbors_mul.get(id).unwrap();

    let key = spatial_idx_rev.get(id).unwrap();
    let new_key = pos.div_euclid(IVec2::splat(config.grid_res));
    let nxy = new_key - key;
    if nxy.cmplt(IVec2::NEG_ONE).any() || nxy.cmpgt(IVec2::ONE).any() {
        return None;
    }

    let neighbor_id = 3 * (nxy.y + 1) + nxy.x + 1;
    let id = neighbors.get(neighbor_id as usize)?;

    let grid = grid_mul.get_mut(*id).unwrap();
    let ixy = pos - key * config.grid_res;
    let index = config.grid_res * ixy.y + ixy.x;
    let cell = grid.get_mut(index as usize).unwrap();
    Some(cell)
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
    sim.set_update_rect(Vec2::new(0.0, 0.0), Vec2::new(64.0, 64.0));

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
