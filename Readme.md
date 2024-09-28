# Scholar Scribe

This project is a web application that allows users to search for scholarly articles and have them summarized by AI. It uses Elasticsearch for efficient search and OpenAI for generating summaries.

## Features

- 


## How to Setup Vast.ai Server

- Create the Server

- select a docker template with GPU. I use Pythorch one with devel image.

- get the ssh connection command from the instance details page.

- Add the following to your `~/.ssh/config` file:
```bash
Host vast-gpu
    HostName <host from instance details page>
    User root
    Port <port from instance details page>
    LocalForward 8080 localhost:8080
```

- connect to the server from vscode.

- Install packages:
```bash
pip install -r requirements.txt
```

- Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key
```

- Generate ssh-key
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

- Add your SSH key to the SSH agent:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
```

- get the ssh-key
```bash
cat ~/.ssh/id_rsa.pub
```

- Add the ssh-key to Github.

- Clone the repository:
```bash
git clone https://github.com/kargarisaac/scholar-scribe.git
```

## Elastic Search as the database

- Run the Elasticsearch container:
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

- If you use mac, use the Arm version of Elasticsearch:
```bash
docker run --platform linux/arm64 -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3-arm64
```


# TODO:

- add the pdf metadata to the elastic search index.
- add the pdf entities to the elastic search index.
- add the pdf summary to the elastic search index.
- add the pdf keywords to the elastic search index.
- add the pdf title to the elastic search index.
- add the pdf author to the elastic search index.
- add the pdf subject to the elastic search index.
- add the pdf creation date to the elastic search index.
- Multimodal RAG for text, image, and table data.
- 
