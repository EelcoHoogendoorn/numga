# Variable Naming Guidelines

- **Avoid cryptic single-letter names for domain quantities**: Do not use single-letter variables for physical or mathematical parameters and domain state (e.g., use `spring_constants` instead of `k`, `stiffness` instead of `s`, `line_extra` instead of `l`).
- **Loop variables are fine**: Standard index/loop iteration variables (`i`, `j`, `k` in numerical loops, or obvious short comprehension variables) are acceptable where context is clear.
- **Exercise judgement**: Use sound engineering judgement and loop back with the user when in doubt.
