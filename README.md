# mbarc-map
Minimally-blocking, Atomic Reference Counted Map

[![Crates.io][crates-badge]][crates-url]
[![MIT licensed][mit-badge]][mit-url]

[crates-badge]: https://img.shields.io/crates/v/mbarc-map.svg
[crates-url]: https://crates.io/crates/mbarc-map
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: https://github.com/gentlecolts/mbarc-map/blob/main/LICENSE

[API Docs](https://docs.rs/mbarc-map/latest/mbarc_map/)

The motivation of the map implemented in this crate is to provide a map that is better suited towards concurrent use.  This crate attempts to solve two problems:
- Need to be able to refer to map elements without keeping the map itself locked
- Data stored within the map should be stored in a way that is as cache-friendly as possible for efficient iteration

Individually, these aren't huge asks, but achieving both of these properties while satisfying rust ended up being complex enough to be worth wrapping, thus leading to the creation of MbarcMap.
You can kind of think of `MbarcMap<T,U>` as a shorthand for: `Mutex<HashMap<T,Arc<Mutex<U>>>>`, however there's more to it than that, especially in regard to pointer safety (stored values are never moved), memory layout (data is stored in continuous blocks), and iterators (safe to alter the map while iterating over it)
