sync:
    cd py-phasmix && uv sync

benchmark:
	cd py-phasmix && uv run scripts/smoke.py

build: sync
	cd py-phasmix && uv run maturin develop --release

build-sse: sync
	cd py-phasmix && RUSTFLAGS="-C target-feature=+sse,+sse2" uv run maturin develop --release

build-avx2: sync
	cd py-phasmix && RUSTFLAGS="-C target-feature=+sse,+sse2,+avx,+avx2" uv run maturin develop --release

build-native: sync
	cd py-phasmix && RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release

clean:
	rm -rf py-phasmix/target target
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	
