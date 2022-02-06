mod mqtt;

use mqtt::Broker;

#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() -> Result<(), String> {
  let broker = Broker::new();
  broker.run().await
}
