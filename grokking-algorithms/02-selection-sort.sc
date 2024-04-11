//> using scala "3"
//> using dep "org.scalacheck::scalacheck::1.17.0"

import org.scalacheck.*
import org.scalacheck.Prop.*

def findSmallest(elems: Vector[Int]): Int =
  elems.zipWithIndex.reduce { case ((a, ia), (b, ib)) =>
    if a < b then (a, ia) else (b, ib)
  }._2

println(findSmallest(Vector(2)))
println(findSmallest((0 to 10).toVector))
println(findSmallest((10 to 0 by -1).toVector))

val distinctVectorGen =
  Gen.listOf(Arbitrary.arbitrary[Int]).map(_.toSet.toVector)
val propFindSmallestSuccess =
  Prop.forAll(distinctVectorGen) { (elems: Vector[Int]) =>
    (elems.nonEmpty) ==> (findSmallest(elems) == elems.indexOf(elems.min))
  }
propFindSmallestSuccess.check()

def selectionSort(elems: Vector[Int]): Vector[Int] =
  elems.zipWithIndex
    .foldLeft[(Vector[Int], Vector[Int])](
      (elems, Vector.empty)
    ) { case (((remaining, acc), _)) =>
      val i = findSmallest(remaining)
      (remaining.zipWithIndex.filter(_._2 != i).map(_._1), acc :+ remaining(i))
    }
    ._2

val elems = Vector(3, 5, 6, 2, 4, 5, 8, 3)
println(selectionSort(elems))

val propSelectionSortSuccess =
  Prop.forAll { (elems: Vector[Int]) =>
    (elems.nonEmpty) ==> (selectionSort(elems) == elems.sorted)
  }
propSelectionSortSuccess.check()
