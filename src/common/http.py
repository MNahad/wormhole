# Copyright 2024 Mohammed Nawabuddin
# SPDX-License-Identifier: Apache-2.0

import http.client


class Http:
    connection: http.client.HTTPSConnection

    def __init__(self, host: str) -> None:
        self.connection = http.client.HTTPSConnection(host)

    def get(self, http_path: str, destination_file: str) -> None:
        self.connection.request("GET", http_path)
        res = self.connection.getresponse()
        body = res.read()
        if res.status // 100 == 2:
            with open(destination_file, "wb") as f:
                f.write(body)
        else:
            print(f"ERROR: PATH {http_path} STATUS {res.status}")

    def close(self) -> None:
        self.connection.close()
