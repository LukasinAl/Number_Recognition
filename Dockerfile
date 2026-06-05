FROM alpine:latest as compile
RUN apk add --no-cache make cmake gcc bash
WORKDIR /build
COPY . .  
RUN ./build.sh


FROM ubuntu:latest
WORKDIR training
RUN mkdir weights
COPY data/ data/
COPY --from compile /build/train_net train_net
ENTRYPOINT ["./train_net"]
