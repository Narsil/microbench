use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, task::JoinSet};

type RequestChannel = async_channel::Receiver<Request>;
type InfoChannel = mpsc::UnboundedSender<ResponseInfo>;

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct Request {
    messages: Vec<Message>,
    stream: bool,
    max_tokens: Option<usize>,
    temperature: f32,
    stream_options: StreamOptions,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Delta {
    // role: String,
    content: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Choice {
    delta: Delta,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Chunk {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Conv {
    // from: String,
    value: String,
}

#[derive(Deserialize)]
struct Conversation {
    conversations: Vec<Conv>,
}

impl From<&Conversation> for Request {
    fn from(value: &Conversation) -> Self {
        Self {
            messages: vec![Message {
                role: "user".to_string(),
                content: value.conversations[0].value.clone(),
            }],
            stream: true,
            max_tokens: Some(500),
            temperature: 0.,
            stream_options: StreamOptions {
                include_usage: true,
            },
        }
    }
}

#[derive(Debug)]
struct ResponseInfo {
    start: std::time::Instant,
    first_token: Option<std::time::Duration>,
    decoded_tokens: usize,
    last_token: Option<std::time::Duration>,
}

impl ResponseInfo {
    fn new() -> Self {
        let start = std::time::Instant::now();
        Self {
            start,
            first_token: None,
            last_token: None,
            decoded_tokens: 0,
        }
    }

    fn update_chunk(&mut self, _chunk: &Chunk) {
        self.decoded_tokens += 1;
        if self.first_token.is_none() {
            self.first_token = Some(self.start.elapsed());
        }
    }

    fn finish(&mut self) {
        self.last_token = Some(self.start.elapsed());
    }
}

impl std::fmt::Display for ResponseInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Result: Prefill {:?} - Decoded {:?} ({} tokens)",
            self.first_token,
            self.last_token.map(|l| l / self.decoded_tokens as u32),
            self.decoded_tokens,
        )
    }
}

async fn run(rchan: RequestChannel, res_tx: InfoChannel) -> Result<()> {
    let client = reqwest::Client::new();

    while let Ok(request) = rchan.recv().await {
        let mut result = ResponseInfo::new();
        let mut response = client
            .post("http://localhost:8080/v1/chat/completions")
            .json(&request)
            .send()
            .await?;
        while let Some(chunk) = response.chunk().await? {
            if *chunk == *b"data: [DONE]\n\n" {
                continue;
            }
            if let Some(chunk) = chunk.strip_prefix(b"data: ") {
                let chunk: Chunk = serde_json::from_slice(chunk).unwrap_or_else(|_| {
                    panic!(
                        "We should parse the chunks {:?}",
                        std::str::from_utf8(chunk)
                    )
                });
                result.update_chunk(&chunk);
            }
        }
        result.finish();
        res_tx.send(result)?
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let virtual_users = 100;
    let n_requests = 200;
    let mut set = JoinSet::new();

    let (tx, rx) = async_channel::bounded(32);
    let (res_tx, mut res_rx) = mpsc::unbounded_channel();

    tokio::task::spawn(async move {
        let data: Vec<Conversation> =
            serde_json::from_str(&std::fs::read_to_string("small.json").unwrap()).unwrap();
        for i in 0..n_requests {
            let conv = &data[i % data.len()];
            let request: Request = conv.into();
            let _ = tx.send(request).await;
        }
    });

    let start = std::time::Instant::now();
    for _ in 0..virtual_users {
        let rx = rx.clone();
        let rtx = res_tx.clone();
        set.spawn(async move { run(rx, rtx).await });
    }
    // XX Very important to drop our own copy of the sender otherwise the receiver never
    // closes.
    drop(res_tx);
    let mut n = 0;
    let mut total_tokens = 0;
    let mut total_time = std::time::Duration::new(0, 0);
    while let Some(result) = res_rx.recv().await {
        n += 1;
        total_tokens += result.decoded_tokens;
        total_time += result.last_token.unwrap();
    }
    set.join_all().await;

    let elapsed_time = start.elapsed();
    println!(
        "Generated {total_tokens} in {total_time:?} ({:?} tok/s/u)",
        total_time / total_tokens as u32 / virtual_users
    );
    println!(
        "Generated {total_tokens} in {elapsed_time:?} ({:?} tok/s)",
        elapsed_time / total_tokens as u32
    );
    assert_eq!(n, n_requests);
    Ok(())
}
