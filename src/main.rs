use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, task::JoinSet};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short, long, default_value_t = 100)]
    virtual_users: usize,

    #[arg(short, long, default_value_t = 200)]
    n_requests: usize,

    #[arg(short, long, default_value_t = 500)]
    max_tokens: usize,

    #[arg(short, long)]
    port: usize,

    #[arg(short, long, action)]
    no_stream: bool,
}

type RequestChannel = async_channel::Receiver<Request>;
type InfoChannel = mpsc::Sender<ResponseInfo>;

#[derive(Deserialize, Serialize, Debug)]
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
    seed: usize,
    stream_options: StreamOptions,
    model: String,
}

#[derive(Deserialize, Debug)]
struct Delta {
    // role: String,
    content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    delta: Delta,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    message: Message,
}

#[derive(Deserialize, Debug)]
struct Usage {
    // prompt_tokens: u32,
    completion_tokens: u32,
    // total_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct Chat {
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Deserialize)]
struct Event {
    choices: Vec<Choice>,
    // usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Error {
    message: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Chunk {
    Event(Event),
    Error(Error),
}

#[derive(Deserialize, Debug)]
struct Conv {
    // from: String,
    value: String,
}

#[derive(Deserialize, Debug)]
struct Conversation {
    conversations: Vec<Conv>,
}

impl Request {
    fn from(value: &Conversation, max_tokens: usize) -> Self {
        Self::_from(value, max_tokens, true)
    }
    fn no_stream(value: &Conversation, max_tokens: usize) -> Self {
        Self::_from(value, max_tokens, false)
    }
    fn _from(value: &Conversation, max_tokens: usize, stream: bool) -> Self {
        log::debug!("{:?}", value.conversations[0].value);
        Self {
            messages: vec![Message {
                role: "user".to_string(),
                // content: value.conversations[0].value.clone(),
                content: "Write an long essay on Deep Learning.".to_string(),
            }],
            model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            stream,
            max_tokens: Some(max_tokens),
            temperature: 0.,
            seed: 0,
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
    text: String,
}

impl ResponseInfo {
    fn new() -> Self {
        let start = std::time::Instant::now();
        Self {
            start,
            first_token: None,
            last_token: None,
            decoded_tokens: 0,
            text: String::new(),
        }
    }

    fn update_chunk(&mut self, event: &Event) {
        if let Some(s) = &event.choices[0].delta.content {
            self.text.push_str(s);
        }
        self.decoded_tokens += 1;
        if self.first_token.is_none() {
            self.first_token = Some(self.start.elapsed());
        }
    }

    fn finish(&mut self) {
        self.last_token = Some(self.start.elapsed());
    }

    fn finish_nostream(&mut self, chat: &Chat) {
        self.decoded_tokens = chat.usage.completion_tokens as usize;
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

async fn run_nostream(port: usize, rchan: RequestChannel, res_tx: InfoChannel) -> Result<()> {
    let client = reqwest::Client::new();

    let url = format!("http://localhost:{port}/v1/chat/completions");
    while let Ok(request) = rchan.recv().await {
        let mut result = ResponseInfo::new();
        let response = client.post(&url).json(&request).send().await?;
        let res: Chat = response.json().await.expect("Valid output");
        log::debug!("{:?}", res.choices[0].message);
        result.finish_nostream(&res);
        res_tx.send(result).await?
    }
    Ok(())
}

async fn run(port: usize, rchan: RequestChannel, res_tx: InfoChannel) -> Result<()> {
    let client = reqwest::Client::new();

    let url = format!("http://localhost:{port}/v1/chat/completions");
    while let Ok(request) = rchan.recv().await {
        let mut result = ResponseInfo::new();
        let mut response = client.post(&url).json(&request).send().await?;
        while let Some(chunk) = response.chunk().await? {
            if *chunk == *b"data: [DONE]\n\n" {
                break;
            }
            if *chunk == *b":\n\n" {
                continue;
            }
            if let Some(chunk) = chunk.strip_prefix(b"data: ") {
                let chunk: Chunk = serde_json::from_slice(chunk).unwrap_or_else(|_| {
                    panic!(
                        "We should parse the chunks {:?}",
                        std::str::from_utf8(chunk)
                    )
                });
                match chunk {
                    Chunk::Event(event) => {
                        if event.choices.is_empty() {
                            continue;
                        }
                        result.update_chunk(&event);
                    }
                    Chunk::Error(err) => {
                        println!("Error {:?}", err.message);
                    }
                }
            } else {
                panic!("Unexpected chunk {:?}", std::str::from_utf8(&chunk));
            }
        }
        result.finish();
        res_tx.send(result).await?
    }
    Ok(())
}

// #[tokio::main]
#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let Cli {
        virtual_users,
        n_requests,
        max_tokens,
        port,
        no_stream,
    } = cli;
    let streaming = !no_stream;
    let mut set = JoinSet::new();

    let (tx, rx) = async_channel::bounded(32);
    let (res_tx, mut res_rx) = mpsc::channel(32);

    tokio::task::spawn(async move {
        let data: Vec<Conversation> =
            serde_json::from_str(&std::fs::read_to_string("small.json").unwrap()).unwrap();
        for i in 0..n_requests {
            let conv = &data[i % data.len()];

            let request: Request = if streaming {
                Request::from(conv, max_tokens)
            } else {
                Request::no_stream(conv, max_tokens)
            };
            let _ = tx.send(request).await;
        }
    });

    let start = std::time::Instant::now();
    for _ in 0..virtual_users {
        let rx = rx.clone();
        let rtx = res_tx.clone();
        set.spawn(async move {
            if streaming {
                run(port, rx, rtx).await
            } else {
                run_nostream(port, rx, rtx).await
            }
        });
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
        // println!("{}", result.text);
    }
    set.join_all().await;

    let elapsed_time = start.elapsed();
    println!(
        "Generated {total_tokens} in {total_time:?} ({:?}/tok - {:.2} tok/s)  (User vision)",
        total_time / total_tokens as u32,
        total_tokens as f32 / total_time.as_secs_f32()
    );
    println!(
        "Generated {total_tokens} in {elapsed_time:?} ({:?}/tok - {:.2} tok/s) (Server vision)",
        elapsed_time / total_tokens as u32,
        total_tokens as f32 / elapsed_time.as_secs_f32()
    );
    assert_eq!(n, n_requests);
    Ok(())
}
