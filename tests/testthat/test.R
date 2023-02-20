g <- random.graph.game(20, 0.3)
V(g)$node_feat <- sample(c("green", "red"), 20, T)
E(g)$edge_feat <- sample(c("black", "white", "gray"), ecount(g), T)


test_that("Correct outcome format and size for base outcome1",
          {
            skip_if_not_installed("torch")
            skip_if(is.na(tryCatch(cuda_is_available(), error = function(e) NA)))
            outcome1 <- spinner(g, "node", "node_feat", "edge_feat")
            expect_equal(class(outcome1), "list")
            expect_equal(length(outcome1), 8)
            expect_equal(names(outcome1), c("graph", "model_description", "model_summary", "pred_fun", "cv_errors", "summary_errors", "history", "time_log"))
            expect_equal(is.igraph(outcome1$graph), TRUE)
            expect_equal(is.function(outcome1$pred_fun), TRUE)
            expect_equal(is.character(outcome1$model_description), TRUE)
            expect_equal(is.list(outcome1$model_summary), TRUE)
            expect_equal(dim(outcome1$cv_errors), c(3, 4))
            expect_equal(length(outcome1$summary_errors), 3)
            expect_equal(is.ggplot(outcome1$history), TRUE)
            expect_equal(class(outcome1$time_log)[1],"Period")
            expect_equal(dim(outcome1$pred_fun(g)[[1]]), c(20, 3))
          })


test_that("Correct outcome format and size for base outcome2",
          {
            skip_if_not_installed("torch")
            skip_if(is.na(tryCatch(cuda_is_available(), error = function(e) NA)))
            outcome2 <- spinner(g, "edge", "node_feat", "edge_feat")
            expect_equal(class(outcome2), "list")
            expect_equal(length(outcome2), 8)
            expect_equal(names(outcome2), c("graph", "model_description", "model_summary", "pred_fun", "cv_errors", "summary_errors", "history", "time_log"))
            expect_equal(is.igraph(outcome2$graph), TRUE)
            expect_equal(is.function(outcome2$pred_fun), TRUE)
            expect_equal(is.character(outcome2$model_description), TRUE)
            expect_equal(is.list(outcome2$model_summary), TRUE)
            expect_equal(dim(outcome2$cv_errors), c(3, 4))
            expect_equal(length(outcome2$summary_errors), 3)
            expect_equal(is.ggplot(outcome2$history), TRUE)
            expect_equal(class(outcome2$time_log)[1],"Period")
            expect_equal(dim(outcome2$pred_fun(g)[[1]]), c(ecount(g), 5))
          })
