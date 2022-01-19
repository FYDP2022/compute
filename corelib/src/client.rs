use rumqttc::{MqttOptions, AsyncClient, EventLoop};

use std::time::Duration;

pub struct Client {
  pub client: AsyncClient,
  pub eventloop: EventLoop,
}

impl Client {
  pub fn new(id: &str) -> Self {
    let mut mqttoptions = MqttOptions::new(id, "localhost", 1883);
    mqttoptions.set_keep_alive(Duration::from_secs(5));
    let (client, eventloop) = AsyncClient::new(mqttoptions, 10);
    Self {
      client,
      eventloop,
    }
  }
}
