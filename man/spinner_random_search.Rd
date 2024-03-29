% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/main2.R
\name{spinner_random_search}
\alias{spinner_random_search}
\title{spinner_random_search}
\usage{
spinner_random_search(
  n_samp,
  graph,
  target,
  node_labels = NA,
  edge_labels = NA,
  context_labels = NA,
  direction = NULL,
  sampling = NA,
  threshold = 0.01,
  method = NULL,
  node_embedding_size = NULL,
  edge_embedding_size = NULL,
  context_embedding_size = NULL,
  update_order = NULL,
  n_layers = NULL,
  skip_shortcut = NULL,
  forward_layer = NULL,
  forward_activation = NULL,
  forward_drop = NULL,
  mode = NULL,
  optimization = NULL,
  epochs = 100,
  lr = NULL,
  patience = 30,
  weight_decay = NULL,
  reps = 1,
  folds = 2,
  holdout = 0.2,
  verbose = TRUE,
  seed = 42,
  keep = FALSE
)
}
\arguments{
\item{n_samp}{Positive integer. Number of models to be randomly generated sampling the hyper-parameter space.}

\item{graph}{A graph in igraph format (without name index for nodes).}

\item{target}{String. Predicted dimension. Options are: "node", "edge".}

\item{node_labels}{String. Character vector with labels of node features. In case of absent features, default to NA (automatic node embedding with selected method).}

\item{edge_labels}{String. Character vector with labels of edge features. In case of absent features, default to NA (automatic edge embedding with selected method).}

\item{context_labels}{String. Character vector with labels of context features. In case of absent features, default to NA (automatic context embedding with selected method).}

\item{direction}{String. Direction of message propagation. Options are: "from_head", "from_tail". Default to: "from_head".}

\item{sampling}{Positive numeric or integer. In case of huge graph, you can opt for a subgraph. Sampling dimension expressed in absolute value or percentage. Default: NA (no sampling).}

\item{threshold}{Numeric. Below this threshold (calculated on edge density), sampling is done on edges, otherwise on nodes. Default: 0.01.}

\item{method}{String. Embedding method in case of absent features. Options are: "null" (zeroed tensor), "laplacian", "adjacency". Default: "null".}

\item{node_embedding_size}{Integer. Size for node embedding. Default: 5.}

\item{edge_embedding_size}{Integer. Size for edge embedding. Default: 5.}

\item{context_embedding_size}{Integer. Size for node embedding. Default: 5.}

\item{update_order}{String. The order of message passing through nodes (n), edges (e) and context (c) for updating information. Available options are: "enc", "nec", "cen", "ecn", "nce", "cne". Default: "enc".}

\item{n_layers}{Integer. Number of graph net variant layers. Default: 1.}

\item{skip_shortcut}{Logical. Flag for applying skip shortcut after the graph net variant layers. Default: FALSE.}

\item{forward_layer}{Integer. Single integer vector with size for forward net layer. Default: 32 (layers with 32 nodes).}

\item{forward_activation}{String. Single character vector with activation for forward net layer. Available options are: "linear", "relu", "mish", "leaky_relu", "celu", "elu", "gelu", "selu", "bent", "softmax", "softmin", "softsign", "sigmoid", "tanh". Default: "relu".}

\item{forward_drop}{Numeric. Single numeric vector with drop out for forward net layer. Default: 0.3.}

\item{mode}{String. Aggregation method for message passing. Options are: "sum", "mean", "max". Default: "sum".}

\item{optimization}{String. Optimization method. Options are: "adadelta", "adagrad", "rmsprop", "rprop", "sgd", "asgd", "adam".}

\item{epochs}{Positive integer. Default: 100.}

\item{lr}{Positive numeric. Learning rate. Default: 0.01.}

\item{patience}{Positive integer. Waiting time (in epochs) before evaluating the overfit performance. Default: 30.}

\item{weight_decay}{Positive numeric. L2-Regularization weight. Default: 0.001.}

\item{reps}{Positive integer. Number of repeated measures. Default: 1.}

\item{folds}{Positive integer. Number of folds for each repetition. Default: 3.}

\item{holdout}{Positive numeric. Percentage of nodes for testing (edges are computed accordingly). Default: 0.2.}

\item{verbose}{Logical. Default: TRUE}

\item{seed}{Random seed. Default: 42.}

\item{keep}{Logical. Flag to TRUE to keep all the explored models. Default: FALSE.}
}
\value{
This function returns a list including:
\itemize{
\item random_search: summary of the sampled hyper-parameters and average error metrics.
\item best: best model according to overall ranking on all average error metrics (for negative metrics, absolute value is considered).
\item time_log: computation time.
\item all_models: list with all generated models (if keep flagged to TRUE).
}
}
\description{
spinner_random_search is a function for fine-tuning using random search on the hyper-parameter space of spinner (predefined or custom).
}
\references{
https://rpubs.com/giancarlo_vercellino/spinner
}
\author{
Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
}
