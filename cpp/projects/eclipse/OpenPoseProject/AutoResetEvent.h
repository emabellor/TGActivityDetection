/*
 * AutoResetEvent.h
 *
 *  Created on: Mar 3, 2018
 *      Author: mauricio
 */

#ifndef AUTORESETEVENT_H_
#define AUTORESETEVENT_H_

#include <mutex>
#include <condition_variable>

class AutoResetEvent
{
	public:
	explicit AutoResetEvent(bool initial = false);

	void Set();
	void Reset();

	bool WaitOne();

	private:
	AutoResetEvent(const AutoResetEvent&);
	AutoResetEvent& operator=(const AutoResetEvent&); // non-copyable
	bool flag_;
	std::mutex protect_;
	std::condition_variable signal_;
};


#endif /* AUTORESETEVENT_H_ */
