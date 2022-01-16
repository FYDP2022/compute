pub struct Broker;

impl Broker {
  pub fn new() -> Self {
    Broker {}
  }

  pub async fn run(&self) -> Result<(), String> {
    loop {}
  }
}