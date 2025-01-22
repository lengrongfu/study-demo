#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod remote;

use anyhow;
use clap::Parser;
use futures_util::StreamExt;
use remote::nvml;
use std::future;
use tarpc::server::incoming::Incoming;
use tarpc::{
    server::{BaseChannel, Channel},
    tokio_serde::formats::Json,
};
/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 50055)]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("args port is {}", args.port);

    // JSON transport is provided by the json_transport tarpc module. It makes it easy
    // to start up a serde-powered json serialization strategy over TCP.
    let listener = tarpc::serde_transport::tcp::listen("10.20.2.102:50051", Json::default).await?;

    println!("xcuda-server listening on `{}`", listener.local_addr());

    listener
        // Ignore accept errors.
        .filter_map(|r| future::ready(r.ok()))
        .map(BaseChannel::with_defaults)
        .map(|channel| channel.execute(nvml::NewRemoteNvmlServer()))
        // Max 100 channels.
        .buffer_unordered(100)
        .for_each(|_| async {})
        .await;

    Ok(())
}
