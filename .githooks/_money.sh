# What "a real money figure" looks like. Sourced by commit-msg and pre-commit
# so the rule exists once.
#
# Blocked: a thousands-separated amount, or a k/M suffix.
# Allowed: bare small figures — a tile caption reading "+€11" says nothing about
# the size of a book, and blocking those would make the hooks noise.
MONEY_PATTERN='(€|EUR ?)[0-9]{1,3}[.,][0-9]{3}|(€|EUR ?)[0-9]+([.,][0-9]+)?[kKmM]\b'
