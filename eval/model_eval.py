# from timeit import default_timer as timer


# def model_eval(model, loss_fn, train_dataloader, test_dataloader, optimizer, num_epochs=5,):
#     start_time = timer()
#     # Train model_0
#     model_results = train(model=model,
#                           train_dataloader=train_dataloader,
#                           test_dataloader=test_dataloader,
#                           optimizer=optimizer,
#                           loss_fn=loss_fn,
#                           epochs=num_epochs)

#     # End the timer and print out how long it took
#     end_time = timer()
#     print(f"Total training time: {end_time-start_time:.3f} seconds")
#     return model_results
