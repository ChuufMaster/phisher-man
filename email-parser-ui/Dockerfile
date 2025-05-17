FROM node:20-alpine as builder

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Optional if serving from nginx
FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html

# If you want to keep dev container
CMD ["npm", "run", "dev"]
