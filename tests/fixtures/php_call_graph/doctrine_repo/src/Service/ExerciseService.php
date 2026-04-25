<?php
namespace App\Service;

use App\Entity\Exercise;
use Doctrine\ORM\EntityManagerInterface;

class ExerciseService
{
    public function __construct(private readonly EntityManagerInterface $em) {}

    public function chained(): void
    {
        $this->em->getRepository(Exercise::class)->findActive();
    }

    public function viaLocal(): void
    {
        $repo = $this->em->getRepository(Exercise::class);
        $repo->findOneByName("foo");
    }
}
