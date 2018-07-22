from __future__ import print_function

import datetime

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
from hmmlearn import hmm

str2date = lambda x: datetime.datetime.strptime(x.decode("utf-8"), '%Y-%m-%d')

path = '/home/mauricio/CSV/yahoo.csv'
quotes = np.genfromtxt(path, delimiter=',', skip_header=1, converters={0: str2date})  # Lines to skip in header!

# Unpack quotes
dates = np.array([q[0] for q in quotes], dtype=datetime.datetime)
close_v = np.array([q[4] for q in quotes])
volume = np.array([q[6] for q in quotes])[1:]

# Take diff of close value. Note that this makes
# ``len(diff) = len(close_t) - 1``, therefore, other quantities also
# need to be shifted by 1.
diff = np.diff(close_v)
dates = dates[1:]
close_v = close_v[1:]

# Pack diff and volume for training.
X = np.column_stack([diff, volume])

print("fitting to HMM and decoding ...", end="")

# Make an HMM instance and execute fit
model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("done")

print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

fig, axs = plt.subplots(model.n_components + 1, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

# plot
ax = axs[model.n_components]
ax.plot_date(dates[:], close_v[:], ".-", c='green')
ax.set_title("Fixed")

# Format the ticks.
ax.xaxis.set_major_locator(YearLocator())
ax.xaxis.set_minor_locator(MonthLocator())

ax.grid(True)


plt.show()
