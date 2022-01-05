use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(
        &["optur/proto/search_space.proto", "optur/proto/study.proto"],
        &["."],
    )?;
    Ok(())
}
