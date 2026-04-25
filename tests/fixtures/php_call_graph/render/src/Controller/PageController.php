<?php
namespace App\Controller;

class PageController
{
    public function index(): object
    {
        return $this->render('page/index.html.twig', ['title' => 'home']);
    }

    public function api(): string
    {
        return $this->renderView('page/api.html.twig');
    }
}
