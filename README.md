To set up, make a directory that all users can access and set `lock_base_directory` at the top of `reserve.py`.
Then create a directory inside that directory with the hostname of the computer (the name printed when you run `hostname` on the command line).
Finally, in the inner directory, create a file called `privileged_users` with the list of usernames that can preempt other users when no devices are free.
