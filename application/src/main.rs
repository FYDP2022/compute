use server::mqtt::Broker;
use vision::VSLAM;

#[async_std::main]
async fn main() {
  let broker = Broker::new();
  let vslam = VSLAM::new();
  futures::try_join!(
    async_std::task::spawn(async move { broker.run().await }),
    async_std::task::spawn(async move { vslam.run().await })
  ).unwrap();
}
