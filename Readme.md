# Scholar Scribe

This project is a web application that allows users to search for scholarly articles and have them summarized by AI. It uses Elasticsearch for efficient search and OpenAI for generating summaries.

## Features

- Search for scholarly articles using Elasticsearch.
- Generate summaries of the articles using OpenAI.
- Save and manage your search history.


# How to Setup Vast.ai Server

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

