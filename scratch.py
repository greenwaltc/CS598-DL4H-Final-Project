self.deepr = DeeprLayer(
            feature_size=self.embedding_dim,
            hidden_size=self.embedding_dim
        )


self.retain = RETAINLayer(
            feature_size=self.embedding_dim,
            dropout=dropout
        )

self.gru = nn.GRU(
    input_size=embedding_dim,
    hidden_size=embedding_dim,
    num_layers=1,
    batch_first=True,
    bidirectional=self.bidirectional,
    dropout=dropout
)