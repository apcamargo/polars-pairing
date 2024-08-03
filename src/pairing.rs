use num_integer::Roots;

pub fn compute_hagen_pair(l: u64, r: u64) -> Option<u64> {
    let (shell, step) = if l > r { (l, r) } else { (r, l) };
    let flag = match (shell % 2, step) {
        (0, s) if s == l => 0,
        (1, s) if s == r => 0,
        _ => 1,
    };
    Some(shell * shell + step * 2 + flag)
}

pub fn compute_hagen_unpair(p: u64) -> Option<(u64, u64)> {
    let shell = p.sqrt();
    let step = (p - shell.pow(2)) / 2;
    if p % 2 == 0 {
        Some((step, shell))
    } else {
        Some((shell, step))
    }
}

pub fn compute_szudzik_pair(l: u64, r: u64) -> Option<u64> {
    if l != l.max(r) {
        Some(r.pow(2) + l)
    } else {
        Some(l.pow(2) + l + r)
    }
}

pub fn compute_szudzik_unpair(p: u64) -> Option<(u64, u64)> {
    let sqrt_p = p.sqrt();
    if p - sqrt_p.pow(2) < sqrt_p {
        Some((p - sqrt_p.pow(2), sqrt_p))
    } else {
        Some((sqrt_p, p - sqrt_p.pow(2) - sqrt_p))
    }
}

pub fn compute_cantor_pair(l: u64, r: u64) -> Option<u64> {
    Some(((l + r) * (l + r + 1) / 2) + r)
}

pub fn compute_cantor_unpair(p: u64) -> Option<(u64, u64)> {
    let w = ((8 * p + 1).sqrt() - 1) / 2;
    let t = (w * (w + 1)) / 2;
    let r = p - t;
    let l = w - r;
    Some((l, r))
}
