mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"frederic_91@orange.fr\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml