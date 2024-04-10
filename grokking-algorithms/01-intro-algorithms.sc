//> using scala "3"
//> using dep "org.scalacheck::scalacheck::1.17.0"

import org.scalacheck.*

def binarySearch(elems: Seq[Int], elem: Int): Option[Int] = {
  def search(start: Int, end: Int): Option[Int] =
    if start > end then None
    else
      val i: Int = (start + end) / 2

      if elems(i) == elem then Some(i)
      else if elems(i) > elem then search(start, i - 1)
      else search(i + 1, end)

  search(0, elems.length - 1)
}

val elems = (0 to 1000).toVector

println(binarySearch(elems, 100))
println(binarySearch(elems, 10000))

val validGen = for {
  end <- Gen.choose(0, 10000)
  elem <- Gen.choose(0, end)
} yield (end, elem)

val propBinarySearchSuccess = Prop.forAll(validGen) { (end, elem) =>
  val elems = (0 to end).toVector
  binarySearch(elems, elem) == Some(elem)
}
propBinarySearchSuccess.check()

val notFoundGen = for {
  end <- Gen.choose(0, 10000)
  elem <- Gen.choose(end + 1, 20000)
} yield (end, elem)

val propBinarySearchNotFound = Prop.forAll(notFoundGen) { (end, elem) =>
  val elems = (0 to end).toVector
  binarySearch(elems, elem) == None
}
propBinarySearchNotFound.check()
