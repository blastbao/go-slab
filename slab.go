//  Copyright 2013-Present Couchbase, Inc.
//
//  Use of this software is governed by the Business Source License included
//  in the file licenses/BSL-Couchbase.txt.  As of the Change Date specified
//  in that file, in accordance with the Business Source License, use of this
//  software will be governed by the Apache License, Version 2.0, included in
//  the file licenses/APL2.txt.

// Package slab provides a 100% golang slab allocator for byte slices.
package slab

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// 在 Arena 内部，内存是按 slab 进行管理的。每个 slab 是一个大块的内存区域，slab 内部连续划分为若干个大小相同的 chunk 。

// Slab: 包含多个 chunk，chunk 是 slab 中的基本内存单元。
// Buf: 从某个 chunk 中分配出来的内存块。
// Footer: 存储在 buf 末尾的元数据，其中包含 slabClassIndex、slabIndex 和 slabMagic 等信息。

// Loc
// An opaque reference to bytes managed by an Arena.  See
// Arena.BufToLoc/LocToBuf().  A Loc struct is GC friendly in that a
// Loc does not have direct pointer fields into the Arena's memory
// that the GC's scanner must traverse.
//
// Loc 是一个轻量级的数据结构，用于标识在 Arena 内存池中的一个特定位置。
// 它包含了用于定位和描述内存块的信息，但不直接引用 Arena 的内存，从而使得 Loc 在垃圾回收（GC）过程中是友好的。
type Loc struct {
	slabClassIndex int // 指向 Arena 中的 slabClass
	slabIndex      int // 指向 slabClass 中的具体 slab
	chunkIndex     int // 指向 slab 中的具体 chunk
	bufStart       int // 在 chunk 中的起始偏移量
	bufLen         int // 表示有效数据的长度
}

// NilLoc returns a Loc where Loc.IsNil() is true.
func NilLoc() Loc {
	return nilLoc
}

var nilLoc = Loc{-1, -1, -1, -1, -1} // A sentinel.

// IsNil returns true if the Loc came from NilLoc().
func (cl Loc) IsNil() bool {
	return cl.slabClassIndex < 0 &&
		cl.slabIndex < 0 &&
		cl.chunkIndex < 0 &&
		cl.bufStart < 0 &&
		cl.bufLen < 0
}

// Slice returns a Loc that a represents a different slice of the
// backing buffer, where the bufStart and bufLen are relative to the
// backing buffer.  Does not change the ref-count of the underlying
// buffer.
//
// NOTE: Many API's (such as BufToLoc) do not correctly handle Loc's
// with non-zero bufStart, so please be careful with using sliced
// Loc's.
func (cl Loc) Slice(bufStart, bufLen int) Loc {
	rv := cl // Makes a copy.
	rv.bufStart = bufStart
	rv.bufLen = bufLen
	return rv
}

// An Arena manages a set of slab classes and memory.
type Arena struct {
	// 控制 slabClass 的增长因子
	growthFactor float64
	// 包含多个 slabClass 实例，每个实例管理不同大小的内存块
	slabClasses []slabClass // slabClasses's chunkSizes grow by growthFactor.
	// 一个魔数，用于验证内存块的合法性
	slabMagic int32 // Magic # suffix on each slab memory []byte.
	// 每个 slab 的大小
	slabSize int
	// 分配内存的函数（可自定义）
	malloc func(size int) []byte // App-specific allocator.

	// 记录内存分配、引用管理和错误情况的各种统计信息
	totAllocs           int64
	totAddRefs          int64
	totDecRefs          int64
	totDecRefZeroes     int64 // Inc'ed when a ref-count reaches zero.
	totGetNexts         int64
	totSetNexts         int64
	totMallocs          int64
	totMallocErrs       int64
	totTooBigErrs       int64
	totAddSlabErrs      int64
	totPushFreeChunks   int64 // Inc'ed when chunk added to free list.
	totPopFreeChunks    int64 // Inc'ed when chunk removed from free list.
	totPopFreeChunkErrs int64
}

type slabClass struct {
	// 包含多个 slab 实例
	slabs []*slab // A growing array of slabs.
	// 每个 chunk 的大小
	chunkSize int // Each slab is sliced into fixed-sized chunks.
	// 空闲 chunk 的链表
	chunkFree Loc // Chunks are tracked in a free-list per slabClass.

	// 总的 chunk 数量
	numChunks int64
	// 空闲 chunk 数量
	numChunksFree int64
}

// slab 是内存块的容器，将内存分成多个 chunk，并包含 chunk 的元数据。
type slab struct {
	// len(memory) == slabSize + slabMemoryFooterLen.
	// 实际的内存数据
	memory []byte
	// Matching array of chunk metadata, and len(memory) == len(chunks).
	// chunk 元数据
	chunks []chunk
}

// Based on slabClassIndex + slabIndex + slabMagic.
const slabMemoryFooterLen int = 4 + 4 + 4

// 内存块的基本单元，具有引用计数和链表信息
type chunk struct {
	// 引用计数
	refs int32 // Ref-count.
	// 指向 chunk 自身的 Loc
	self Loc // The self is the Loc for this chunk.
	// 指向下一个 chunk 的 Loc，用于链表
	next Loc // Used when chunk is in the free-list or when chained.
}

// NewArena returns an Arena to manage byte slice memory based on a
// slab allocator approach.
//
// The startChunkSize and slabSize should be > 0.
// The growthFactor should be > 1.0.
// The malloc() func is invoked when Arena needs memory for a new slab.
// When malloc() is nil, then Arena defaults to make([]byte, size).
//
// 创建一个新的 Arena 实例
func NewArena(
	startChunkSize int,
	slabSize int,
	growthFactor float64,
	malloc func(size int) []byte) *Arena {
	if malloc == nil {
		malloc = defaultMalloc
	}
	s := &Arena{
		growthFactor: growthFactor,
		slabMagic:    rand.Int31(),
		slabSize:     slabSize,
		malloc:       malloc,
	}
	s.addSlabClass(startChunkSize)
	return s
}

func defaultMalloc(size int) []byte {
	return make([]byte, size)
}

// Alloc may return nil on errors, such as if no more free chunks are
// available and new slab memory was not allocatable (such as if
// malloc() returns nil).  The returned buf may not be append()'ed to
// for growth.  The returned buf must be DecRef()'ed for memory reuse.
//
// 分配内存块，并返回指向内存块的字节切片
func (s *Arena) Alloc(bufLen int) (buf []byte) {
	sc, chunk := s.allocChunk(bufLen)
	if sc == nil || chunk == nil {
		return nil
	}
	return sc.chunkMem(chunk)[0:bufLen]
}

// Owns returns true if this Arena owns the buf.
//
// 检查一个字节切片是否属于该 Arena
func (s *Arena) Owns(buf []byte) bool {
	sc, c := s.bufChunk(buf)
	return sc != nil && c != nil
}

// AddRef increase the ref count on a buf.  The input buf must be from
// an Alloc() from the same Arena.
//
// 管理内存块的引用计数
func (s *Arena) AddRef(buf []byte) {
	s.totAddRefs++
	sc, c := s.bufChunk(buf)
	if sc == nil || c == nil {
		panic("buf not from this arena")
	}
	c.addRef()
}

// DecRef decreases the ref count on a buf.  The input buf must be
// from an Alloc() from the same Arena.  Once the buf's ref-count
// drops to 0, the Arena may reuse the buf.  Returns true if this was
// the last DecRef() invocation (ref count reached 0).
//
// 管理内存块的引用计数
func (s *Arena) DecRef(buf []byte) bool {
	s.totDecRefs++
	sc, c := s.bufChunk(buf)
	if sc == nil || c == nil {
		panic("buf not from this arena")
	}
	return s.decRef(sc, c)
}

// GetNext returns the next chained buf for the given input buf.  The
// buf's managed by an Arena can be chained.  The returned bufNext may
// be nil.  When the returned bufNext is non-nil, the caller owns a
// ref-count on bufNext and must invoke DecRef(bufNext) when the
// caller is finished using bufNext.
//
// 管理内存块的链表关系
func (s *Arena) GetNext(buf []byte) (bufNext []byte) {
	s.totGetNexts++
	sc, c := s.bufChunk(buf)
	if sc == nil || c == nil {
		panic("buf not from this arena")
	}
	if c.refs <= 0 {
		panic(fmt.Sprintf("unexpected ref-count during GetNext: %#v", c))
	}

	scNext, cNext := s.chunk(c.next)
	if scNext == nil || cNext == nil {
		return nil
	}

	cNext.addRef()

	return scNext.chunkMem(cNext)[c.next.bufStart : c.next.bufStart+c.next.bufLen]
}

// SetNext associates the next chain buf following the input buf to be
// bufNext.  The buf's from an Arena can be chained, where buf will
// own an AddRef() on bufNext.  When buf's ref-count goes to zero, it
// will call DecRef() on bufNext.  The bufNext may be nil.  The
// bufNext must have start position 0 (or bufStart of 0) with respect
// to its backing buffer.
func (s *Arena) SetNext(buf, bufNext []byte) {
	s.totSetNexts++
	sc, c := s.bufChunk(buf)
	if sc == nil || c == nil {
		panic("buf not from this arena")
	}
	if c.refs <= 0 {
		panic(fmt.Sprintf("refs <= 0 during SetNext: %#v", c))
	}

	scOldNext, cOldNext := s.chunk(c.next)
	if scOldNext != nil && cOldNext != nil {
		s.decRef(scOldNext, cOldNext)
	}

	c.next = nilLoc
	if bufNext != nil {
		scNewNext, cNewNext := s.bufChunk(bufNext)
		if scNewNext == nil || cNewNext == nil {
			panic("bufNext not from this arena")
		}
		cNewNext.addRef()

		c.next = cNewNext.self
		c.next.bufStart = 0
		c.next.bufLen = len(bufNext)
	}
}

// BufToLoc returns a Loc that represents an Arena-managed buf.  Does
// not affect the reference count of the buf.  The buf slice must have
// start position 0 (must not be a sliced Loc with non-zero bufStart).
//
// 在字节切片和 Loc 之间转换
func (s *Arena) BufToLoc(buf []byte) Loc {
	sc, c := s.bufChunk(buf)
	if sc == nil || c == nil {
		return NilLoc()
	}

	var loc = c.self // Makes a copy.
	loc.bufStart = 0
	loc.bufLen = len(buf)
	return loc
}

// LocToBuf returns a buf for an Arena-managed Loc.  Does not affect
// the reference count of the buf.  The Loc may have come from
// Loc.Slice().
func (s *Arena) LocToBuf(loc Loc) []byte {
	sc, chunk := s.chunk(loc)
	if sc == nil || chunk == nil {
		return nil
	}
	return sc.chunkMem(chunk)[loc.bufStart : loc.bufStart+loc.bufLen]
}

func (s *Arena) LocAddRef(loc Loc) {
	s.totAddRefs++
	sc, chunk := s.chunk(loc)
	if sc == nil || chunk == nil {
		return
	}
	chunk.addRef()
}

func (s *Arena) LocDecRef(loc Loc) {
	s.totDecRefs++
	sc, chunk := s.chunk(loc)
	if sc == nil || chunk == nil {
		return
	}
	s.decRef(sc, chunk)
}

// ---------------------------------------------------------------

// 分配一个合适的 chunk，根据 bufLen 找到合适的 slabClass 。
// 如果没有空闲 chunk，则会尝试添加新的 slab。
func (s *Arena) allocChunk(bufLen int) (*slabClass, *chunk) {
	s.totAllocs++

	if bufLen > s.slabSize {
		s.totTooBigErrs++
		return nil, nil
	}

	// 查找适合当前 bufLen 的 slabClass ，slabClass 代表了一组大小相同的 chunk 。
	slabClassIndex := s.findSlabClassIndex(bufLen)
	sc := &(s.slabClasses[slabClassIndex])

	// 接着，检查 slabClass 的 chunkFree 列表是否有可用的 chunk

	// 如果 slabClass 的 chunkFree 列表为空，表示没有可用的 chunk 。
	// 这时，Arena 会调用 addSlab 方法添加新的 slab，从而增加可用的 chunk。
	// 如果 addSlab 失败，则记录错误并返回 nil 。
	if sc.chunkFree.IsNil() {
		if !s.addSlab(slabClassIndex, s.slabSize, s.slabMagic) {
			s.totAddSlabErrs++
			return nil, nil
		}
	}

	// 如果 chunkFree 列表中有空闲的 chunk，则从中弹出一个，并记录成功的分配操作。
	// 如果弹出失败，记录错误并返回 nil。
	s.totPopFreeChunks++
	chunk := sc.popFreeChunk()
	if chunk == nil {
		s.totPopFreeChunkErrs++
		return nil, nil
	}

	// 成功分配，返回 chunk 和 slabClass
	return sc, chunk
}

// 确定适合给定 bufLen 的 slabClass 索引。如果当前没有适合的 slabClass，则创建新的 slabClass
func (s *Arena) findSlabClassIndex(bufLen int) int {
	// 二分查找 slabClasses 列表中第一个满足条件的 slabClass
	i := sort.Search(len(s.slabClasses), func(i int) bool { return bufLen <= s.slabClasses[i].chunkSize })

	// 没有合适的 slabClass ，需要创建一个新的 slabClass
	if i >= len(s.slabClasses) {
		// 获取当前列表中的最后一个 slabClass 。
		slabClass := &(s.slabClasses[len(s.slabClasses)-1])
		// 计算下一个 chunkSize，它是当前 chunkSize 乘以 growthFactor 。
		// growthFactor 是一个大于 1 的因子，用于确定下一个 slabClass 的大小。
		nextChunkSize := float64(slabClass.chunkSize) * s.growthFactor
		// 添加新的 slabClass 。
		// 这里的 nextChunkSize 被向上取整，以确保 chunkSize 是一个整数。
		s.addSlabClass(int(math.Ceil(nextChunkSize)))
		// 再次查找
		return s.findSlabClassIndex(bufLen)
	}

	return i
}

// 添加新的 slabClass，并初始化 chunkSize
func (s *Arena) addSlabClass(chunkSize int) {
	s.slabClasses = append(s.slabClasses, slabClass{
		chunkSize: chunkSize,
		chunkFree: nilLoc,
	})
}

// 为指定的 slabClass 添加新的 slab 。
// 初始化 slab 内存，并将其划分为 chunk 。
func (s *Arena) addSlab(
	slabClassIndex,
	slabSize int,
	slabMagic int32,
) bool {

	// 获取 Slab Class
	sc := &(s.slabClasses[slabClassIndex])

	// 计算每个 Slab 可以包含的 Chunk 数量
	chunksPerSlab := slabSize / sc.chunkSize // chunksPerSlab 表示在给定的 slabSize 下，每个 slab 可以容纳多少个 chunk 。
	if chunksPerSlab <= 0 {                  // 如果 chunksPerSlab 计算结果小于等于 0，则设置为 1，确保每个 slab 至少能容纳一个 chunk 。
		chunksPerSlab = 1
	}

	slabIndex := len(sc.slabs)

	// 分配内存
	s.totMallocs++
	// Re-multiplying to avoid any extra fractional chunk memory.
	memorySize := (sc.chunkSize * chunksPerSlab) + slabMemoryFooterLen
	memory := s.malloc(memorySize)
	if memory == nil {
		s.totMallocErrs++
		return false
	}

	// 初始化 Slab
	slab := &slab{
		memory: memory,
		chunks: make([]chunk, chunksPerSlab),
	}

	// 设置 Slab 的 Footer
	footer := slab.memory[len(slab.memory)-slabMemoryFooterLen:]
	binary.BigEndian.PutUint32(footer[0:4], uint32(slabClassIndex))
	binary.BigEndian.PutUint32(footer[4:8], uint32(slabIndex))
	binary.BigEndian.PutUint32(footer[8:12], uint32(slabMagic))

	// 将 Slab 添加到 SC 中
	sc.slabs = append(sc.slabs, slab)

	// 初始化 Chunk
	for i := 0; i < len(slab.chunks); i++ {
		c := &(slab.chunks[i])
		// 为每个 chunk 设置其属性（如所属的 slab 类别、索引、chunk 的索引等）
		c.self.slabClassIndex = slabClassIndex
		c.self.slabIndex = slabIndex
		c.self.chunkIndex = i
		c.self.bufStart = 0
		c.self.bufLen = sc.chunkSize
		// 调用 sc.pushFreeChunk(c) 将 chunk 标记为“空闲”，准备供以后使用。
		sc.pushFreeChunk(c)
	}

	// 更新统计信息
	sc.numChunks += int64(len(slab.chunks))

	return true
}

// 将一个 chunk 推入空闲链表
func (sc *slabClass) pushFreeChunk(c *chunk) {
	if c.refs != 0 {
		panic(fmt.Sprintf("pushFreeChunk() non-zero refs: %v", c.refs))
	}
	c.next = sc.chunkFree
	sc.chunkFree = c.self
	sc.numChunksFree++
}

// 从空闲链表中弹出一个 chunk
func (sc *slabClass) popFreeChunk() *chunk {
	// 1. 检查空闲链表是否为空
	// 	sc.chunkFree 是 slabClass 内部维护的一个链表，存储着所有空闲的 chunk 。
	// 	如果 chunkFree 为 nilLoc（即链表为空），则表示没有空闲的 chunk。
	if sc.chunkFree.IsNil() {
		panic("popFreeChunk() when chunkFree is nil")
	}

	// 2. 获取 chunk
	//	通过 sc.chunk(sc.chunkFree) 获取当前链表中的第一个 chunk 。
	c := sc.chunk(sc.chunkFree)

	// 3. 检查引用计数
	//	在弹出 chunk 之前，代码检查其引用计数 c.refs 是否为 0 。
	//	chunk 应该是空闲的，因此引用计数应该为 0 ，如果不是，说明这个 chunk 可能被其他地方引用，可能存在数据一致性问题，此时会触发 panic 。
	if c.refs != 0 {
		panic(fmt.Sprintf("popFreeChunk() non-zero refs: %v", c.refs))
	}

	// 4. 设置引用计数
	//	将 chunk 的引用计数 c.refs 设置为 1 ，因为它现在被“弹出”并正在使用。
	//	这样做可以确保 chunk 在被使用期间不会被回收。
	c.refs = 1

	// 5. 更新链表
	//	将 sc.chunkFree 更新为 c.next，即将链表的头指针指向下一个 chunk 。
	//	将 c.next 设置为 nilLoc ，表示这个 chunk 不再在链表中。
	sc.chunkFree = c.next
	c.next = nilLoc

	// 6. 更新空闲 chunk 数量
	//	减少 numChunksFree 计数器的值，表示一个空闲的 chunk 被弹出。
	//	接着，检查 numChunksFree 是否小于 0 。如果小于 0，说明空闲 chunk 的数量出现了不一致的情况，函数会引发 panic 。
	sc.numChunksFree--
	if sc.numChunksFree < 0 {
		panic("popFreeChunk() got < 0 numChunksFree")
	}
	return c
}

// 根据 chunk 的位置从 slab 内存中提取实际数据。
func (sc *slabClass) chunkMem(c *chunk) []byte {

	// 检查有效性：
	//	如果 c 是 nil，说明没有有效的 chunk，直接返回 nil 。
	//	如果 c.self 是 nil，说明 chunk 的元数据无效，也应返回 nil。
	if c == nil || c.self.IsNil() {
		return nil
	}

	// 计算起始位置：
	//	sc.chunkSize：每个 chunk 的大小。
	//	c.self.chunkIndex：chunk 在某个数据结构中的索引。
	//	beg：计算出 chunk 在内存中的起始字节位置，单位是字节。这个位置是通过 chunk 大小和索引乘积得出的。
	beg := sc.chunkSize * c.self.chunkIndex

	// 提取内存区域：
	//	sc.slabs：slabClass 中存储的所有 slab 对象的数组。
	//	c.self.slabIndex：chunk 在 slabs 数组中的位置索引。根据这个索引获取具体的 slab。
	//	sc.slabs[c.self.slabIndex].memory：获取 slab 对象的内存区域，它是一个字节切片 ([]byte)，表示整个 slab 的内存。
	//	beg : beg+sc.chunkSize：使用切片操作提取从 beg 到 beg+sc.chunkSize 的内存区域。这样可以获取到 chunk 在 slab 内存区域中的实际数据部分。
	return sc.slabs[c.self.slabIndex].memory[beg : beg+sc.chunkSize]
}

// 根据 Loc 获取 chunk
func (sc *slabClass) chunk(cl Loc) *chunk {
	if cl.IsNil() {
		return nil
	}
	return &(sc.slabs[cl.slabIndex].chunks[cl.chunkIndex])
}

func (s *Arena) chunk(cl Loc) (*slabClass, *chunk) {
	if cl.IsNil() {
		return nil, nil
	}
	sc := &(s.slabClasses[cl.slabClassIndex])
	return sc, sc.chunk(cl)
}

// Determine the slabClass & chunk for an Arena managed buf []byte.
//
// 解析 buf 的 footer 来确定它所属的 slabClass 和 chunk 。
func (s *Arena) bufChunk(buf []byte) (*slabClass, *chunk) {
	if buf == nil || cap(buf) <= slabMemoryFooterLen {
		return nil, nil
	}

	// 计算 footer 起始位置，位于 buf 的尾部 xx 字节
	rest := buf[:cap(buf)]
	footerDistance := len(rest) - slabMemoryFooterLen
	footer := rest[footerDistance:]

	// footer = slabClassIndex(4B) + slabIndex(4B) + slabMagic(4B)
	// 其中 slabMagic 用于验证数据的合法性。
	slabClassIndex := binary.BigEndian.Uint32(footer[0:4])
	slabIndex := binary.BigEndian.Uint32(footer[4:8])
	slabMagic := binary.BigEndian.Uint32(footer[8:12])
	if slabMagic != uint32(s.slabMagic) {
		return nil, nil
	}

	// 先获取 slabClass ，再获取 slab ，
	slabClass := &(s.slabClasses[slabClassIndex])
	slab := slabClass.slabs[slabIndex]

	// [重要]
	// 为了找到 buf 对应的 chunk，需要确定 buf 在 slab 中的位置。
	//
	// footerDistance / slabClass.chunkSize 计算了 footerDistance 涉及的 chunk 数量，可能会得到一个浮点数。
	// math.Ceil 确保即使 footerDistance 不是 chunkSize 的整数倍，也能计算得到整数 chunkIndex 。
	chunkIndex := len(slab.chunks) - int(math.Ceil(float64(footerDistance)/float64(slabClass.chunkSize))) /// ???
	return slabClass, &(slab.chunks[chunkIndex])
}

func (c *chunk) addRef() *chunk {
	c.refs++
	if c.refs <= 1 {
		panic(fmt.Sprintf("refs <= 1 during addRef: %#v", c))
	}
	return c
}

// 减少 chunk 的引用计数。如果引用计数降到 0，将 chunk 推入空闲链表
func (s *Arena) decRef(sc *slabClass, c *chunk) bool {
	c.refs--
	if c.refs < 0 {
		panic(fmt.Sprintf("refs < 0 during decRef: %#v", c))
	}
	if c.refs == 0 {
		s.totDecRefZeroes++
		scNext, cNext := s.chunk(c.next)
		if scNext != nil && cNext != nil {
			s.decRef(scNext, cNext)
		}
		c.next = nilLoc
		s.totPushFreeChunks++
		sc.pushFreeChunk(c)
		return true
	}
	return false
}

// Stats fills an input map with runtime metrics about the Arena.
func (s *Arena) Stats(m map[string]int64) map[string]int64 {
	m["totSlabClasses"] = int64(len(s.slabClasses))
	m["totAllocs"] = s.totAllocs
	m["totAddRefs"] = s.totAddRefs
	m["totDecRefs"] = s.totDecRefs
	m["totDecRefZeroes"] = s.totDecRefZeroes
	m["totGetNexts"] = s.totGetNexts
	m["totSetNexts"] = s.totSetNexts
	m["totMallocs"] = s.totMallocs
	m["totMallocErrs"] = s.totMallocErrs
	m["totTooBigErrs"] = s.totTooBigErrs
	m["totAddSlabErrs"] = s.totAddSlabErrs
	m["totPushFreeChunks"] = s.totPushFreeChunks
	m["totPopFreeChunks"] = s.totPopFreeChunks
	m["totPopFreeChunkErrs"] = s.totPopFreeChunkErrs
	for i, sc := range s.slabClasses {
		prefix := fmt.Sprintf("slabClass-%06d-", i)
		m[prefix+"numSlabs"] = int64(len(sc.slabs))
		m[prefix+"chunkSize"] = int64(sc.chunkSize)
		m[prefix+"numChunks"] = int64(sc.numChunks)
		m[prefix+"numChunksFree"] = int64(sc.numChunksFree)
		m[prefix+"numChunksInUse"] = int64(sc.numChunks - sc.numChunksFree)
	}
	return m
}
