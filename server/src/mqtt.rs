use librumqttd::Config;
use librumqttd::async_locallink::construct_broker;

pub struct Broker;

impl Broker {
  pub fn new() -> Self {
    Broker {}
  }

  pub async fn run(&self) -> Result<(), String> {
    println!("MQTT server starting...");
    let config: Config = toml::from_str(include_str!("config.toml")).unwrap();

    let (mut router, console, servers, builder) = construct_broker(config);

    tokio::spawn(async move {
      router.start().unwrap();
    });

    let (mut tx, mut rx) = builder.connect("localclient", 200).await.unwrap();
    tx.subscribe(std::iter::once("#")).await.unwrap();

    let console_task = tokio::spawn(console);

    let sub_task = tokio::spawn(async move {
      loop {
        let message = rx.recv().await.unwrap();
        // println!("T = {}, P = {:?}", message.topic, message.payload.len());
        println!("T = {}, M = {:#?}", message.topic, message.payload);
      }
    });

    servers.await;
    sub_task.await.unwrap();
    console_task.await.unwrap();
    Ok(())
  }
}