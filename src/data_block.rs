pub(crate) struct DataBlock<T, const BLOCK_SIZE: usize> {
	free_space: usize,
	data: [Option<T>; BLOCK_SIZE],
}

impl<T, const BLOCK_SIZE: usize> Default for DataBlock<T, BLOCK_SIZE> {
	fn default() -> Self {
		Self {
			free_space: BLOCK_SIZE,
			data: [Self::EMPTY_ELEMENT; BLOCK_SIZE],
		}
	}
}

impl<T, const BLOCK_SIZE: usize> DataBlock<T, BLOCK_SIZE> {
	const EMPTY_ELEMENT: Option<T> = None;

	pub(crate) fn insert(&mut self, val: T) -> usize {
		assert!(self.free_space > 0);

		//the above assert means this MUST find a value, assuming free_space is accurate
		let mut stored_offset = 0;
		let last_index = self.data.len() - 1;
		//TODO: improve this loop further.  maybe iters like find?
		for (i, item) in self.data.iter_mut().enumerate() {
			if item.is_none() {
				let _ = item.insert(val);
				stored_offset = i;
				break;
			}
			if i == last_index {
				std::unreachable!();
			}
		}
		self.free_space -= 1;

		stored_offset
	}

	pub(crate) fn remove(&mut self, index: usize) -> Option<T> {
		let removed = self.data.get_mut(index).unwrap().take();

		if removed.is_some() {
			self.free_space += 1;
		}

		removed
	}

	pub(crate) fn free_space(&self) -> usize {
		self.free_space
	}

	pub(crate) unsafe fn get_raw(&mut self, offset: usize) -> Option<*mut T> {
		self.data[offset].as_mut().map(|val| val as *mut T)
	}

	pub(crate) fn total_free_blocks(&self) -> usize {
		self.data
			.iter()
			.filter(|item| -> bool { item.is_none() })
			.count()
	}
}
