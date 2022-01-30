use server::mqtt::Broker;
use vision::VSLAM;

use futures::TryFutureExt;

#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() {
  let broker = Broker::new();
  let vslam = VSLAM::new();
  futures::try_join!(
    broker.run(),
    tokio::spawn(async move { vslam.run().await })
      .unwrap_or_else(|err| Err(err.to_string()))
  ).unwrap();
}
