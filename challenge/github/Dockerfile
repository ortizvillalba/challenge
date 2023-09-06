name: CD

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8

      - name: Install Heroku CLI
        run: |
          curl https://cli-assets.heroku.com/install.sh | sh
          heroku --version

      - name: Login to Heroku
        run: heroku login -i

      - name: Deploy to Heroku
        run: |
          heroku create your-heroku-app-name
          git push heroku main
