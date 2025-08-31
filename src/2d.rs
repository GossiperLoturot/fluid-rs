use std::io::Write;

use crossterm::{style::Stylize, ExecutableCommand};
use glam::*;
use rand::Rng;

const DT: f32 = 0.032;
const ITERATIONS: i32 = (1.0 / DT) as i32;

const PARTICLE_SIZE: i32 = 4096;
const GRID_RES: i32 = 64;

const GRAVITY: Vec2 = Vec2::new(0.0, 0.3);
const REST_DENSITY: f32 = 4.0;
const DYNAMIC_VISCOSITY: f32 = 0.1;
const EOS_STIFFNESS: f32 = 10.0;
const EOS_POWER: f32 = 4.0;

const STDOUT_SIZE: IVec2 = IVec2::new(80, 40);

const ENFORCE_RADIUS: f32 = 10.0;

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

fn g2p(particles: &mut [Particle], grid: &[Cell], enforce: &Option<Vec2>) {
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

        // enforce

        if let Some(enforce_x) = enforce {
            let dist = p.x - enforce_x;
            if dist.length_squared() < ENFORCE_RADIUS * ENFORCE_RADIUS {
                p.v += dist.normalize_or_zero();
            }
        }

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

// console input

#[derive(Clone, Copy)]
enum Event {
    Quit,
    Drag(u16, u16),
    Resize(u16, u16),
}

fn input_handle(tx: crossbeam_channel::Sender<Event>) {
    loop {
        match crossterm::event::read() {
            Ok(event) => match event {
                crossterm::event::Event::Key(key_event) => {
                    if key_event.code == crossterm::event::KeyCode::Char('q') {
                        let _ = tx.send(Event::Quit);
                    }
                }
                crossterm::event::Event::Mouse(mouse_event) => {
                    if matches!(mouse_event.kind, crossterm::event::MouseEventKind::Down(_)) {
                        let _ = tx.try_send(Event::Drag(mouse_event.column, mouse_event.row));
                    }
                    if matches!(mouse_event.kind, crossterm::event::MouseEventKind::Drag(_)) {
                        let _ = tx.try_send(Event::Drag(mouse_event.column, mouse_event.row));
                    }
                }
                crossterm::event::Event::Resize(x, y) => {
                    let _ = tx.send(Event::Resize(x, y));
                }
                _ => {}
            },
            Err(_) => {}
        }
    }
}

// console output

fn draw_base<T: Write + ?Sized>(out: &mut T) -> std::io::Result<()> {
    out.execute(crossterm::terminal::Clear(
        crossterm::terminal::ClearType::All,
    ))?;

    let msg = "2D MLS-MPM Fluid Simulation -- [Drag mouse] Interact, [q] Quit.";
    out.execute(crossterm::cursor::MoveTo(0, STDOUT_SIZE.y as u16))?
        .execute(crossterm::style::Print(msg.dark_green().bold()))?;

    Ok(())
}

fn draw<T: Write + ?Sized>(particles: &[Particle], out: &mut T) -> std::io::Result<()> {
    let mut bin_counts = [0; (STDOUT_SIZE.x * STDOUT_SIZE.y) as usize];

    for i in 0..PARTICLE_SIZE {
        let xyz = (particles[i as usize].x / Vec2::splat(GRID_RES as f32) * STDOUT_SIZE.as_vec2())
            .as_ivec2();

        if xyz.cmpge(IVec2::ZERO).all() && xyz.cmplt(STDOUT_SIZE).all() {
            bin_counts[(xyz.y * STDOUT_SIZE.x + xyz.x) as usize] += 1;
        }
    }

    for y in 0..STDOUT_SIZE.y {
        out.execute(crossterm::cursor::MoveTo(0, y as u16))?;
        for x in 0..STDOUT_SIZE.x {
            let ch = match bin_counts[(y * STDOUT_SIZE.x + x) as usize] {
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

// main routine

fn main() -> std::io::Result<()> {
    let mut rng = rand::rng();

    let time = std::time::Duration::from_secs_f32(DT);

    let mut particles = [Particle::default(); PARTICLE_SIZE as usize];
    let mut grid = [Cell::default(); (GRID_RES * GRID_RES) as usize];

    for i in 0..PARTICLE_SIZE {
        let x = rng.random_range(16.0..=48.0);
        let y = rng.random_range(16.0..=48.0);
        particles[i as usize].x = Vec2::new(x, y);
        particles[i as usize].v = Vec2::ZERO;
        particles[i as usize].c_mat = Mat2::ZERO;
        particles[i as usize].m = 1.0;
    }

    for i in 0..GRID_RES * GRID_RES {
        grid[i as usize].v = Vec2::ZERO;
        grid[i as usize].m = 0.0;
    }

    let mut out = std::io::BufWriter::new(std::io::stdout());

    crossterm::terminal::enable_raw_mode()?;
    out.execute(crossterm::terminal::EnterAlternateScreen)?;
    out.execute(crossterm::cursor::Hide)?;
    out.execute(crossterm::event::EnableMouseCapture)?;

    draw_base(&mut out)?;

    let (tx, rx) = crossbeam_channel::bounded::<Event>(1);
    std::thread::spawn(|| input_handle(tx));
    loop {
        let mut enforce = None;

        match rx.try_recv() {
            Ok(event) => match event {
                Event::Quit => break,
                Event::Drag(x, y) => {
                    let x = x as f32 / STDOUT_SIZE.x as f32 * GRID_RES as f32;
                    let y = y as f32 / STDOUT_SIZE.y as f32 * GRID_RES as f32;
                    enforce = Some(Vec2::new(x, y));
                }
                Event::Resize(width, height) => {
                    if width < STDOUT_SIZE.x as u16 || height < STDOUT_SIZE.y as u16 {
                        break;
                    }
                    draw_base(&mut out)?;
                }
            },
            Err(crossbeam_channel::TryRecvError::Empty) => {}
            Err(crossbeam_channel::TryRecvError::Disconnected) => break,
        }

        draw(&particles, &mut out)?;

        for _ in 0..ITERATIONS {
            clear_grid(&mut grid);
            p2g_1(&particles, &mut grid);
            p2g_2(&particles, &mut grid);
            update_grid(&mut grid);
            g2p(&mut particles, &grid, &enforce);
        }

        std::thread::sleep(time);
    }

    out.execute(crossterm::event::DisableMouseCapture)?;
    out.execute(crossterm::cursor::Show)?;
    out.execute(crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::terminal::disable_raw_mode()?;

    Ok(())
}
