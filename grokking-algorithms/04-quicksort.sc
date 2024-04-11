//> using scala "3"
//> using dep "org.scalacheck::scalacheck::1.17.0"

import org.scalacheck.*
import org.scalacheck.Prop.*

def quicksort(elems: Seq[Int]): Seq[Int] = {
  if (elems.length <= 1) {
    elems
  } else {
    val pivot = elems(0)
    val (lt, gt) = elems.tail.partition(_ <= pivot)
    quicksort(lt) ++ Seq(pivot) ++ quicksort(gt)
  }
}

println(quicksort(Vector(1, 3, 5, 2, 6, 4)))

val propQuicksortSuccess =
  Prop.forAll { (elems: Vector[Int]) =>
    quicksort(elems) == elems.sorted
  }
propQuicksortSuccess.check()
